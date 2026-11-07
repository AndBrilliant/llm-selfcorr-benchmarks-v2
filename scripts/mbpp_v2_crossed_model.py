#!/usr/bin/env python3
"""Crossed evaluator x generator analysis of cross-evaluation acceptance.

For each arm, fits an accepting-incorrect-candidate logistic model with
categorical evaluator and generator effects plus a same-family indicator.
Task-clustered sandwich standard errors account for repeated candidates across
the four external evaluators and for within-task dependence. This separates a
same-family acceptance effect from each evaluator's general permissiveness.

Also emits the full 5 x 5 false-acceptance matrix for each arm.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
V = {"ACCEPT", "REJECT"}
MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
NAMES = {"gpt-5.4": "GPT-5.4", "ds-v4-flash": "DeepSeek", "glm-5.3-flash": "GLM",
         "qwen38-flash": "Qwen", "gpt-6-luna": "GPT-6-Luna"}
OPENAI = {"gpt-5.4", "gpt-6-luna"}


def load_records(arm: str) -> pd.DataFrame:
    rows = []
    for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("T") == 0 and r.get(f"{arm}_verdict") in V:
            rows.append({"evaluator": r["model"], "generator": r["model"],
                         "task_id": r["task_id"], "same": 1,
                         "same_provider": 1,
                         "accept": int(r[f"{arm}_verdict"] == "ACCEPT")})
    for l in (ROOT / "data/mbpp_v2/cross_eval/cross_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get(f"{arm}_verdict") not in V:
            continue
        prod_key = (r["generator"], r["task_id"])
        if prod_key not in PROD or PROD[prod_key].get("T") != 0:
            continue
        rows.append({"evaluator": r["evaluator"], "generator": r["generator"],
                     "task_id": r["task_id"], "same": 0,
                     "same_provider": int(r["evaluator"] in OPENAI and r["generator"] in OPENAI),
                     "accept": int(r[f"{arm}_verdict"] == "ACCEPT")})
    return pd.DataFrame(rows)


def fit_clustered(df: pd.DataFrame, include_provider: bool = False) -> dict:
    x = pd.get_dummies(df[["evaluator", "generator"]], drop_first=True, dtype=float)
    x["same"] = df["same"].astype(float)
    if include_provider:
        x["same_provider"] = df["same_provider"].astype(float)
    x.insert(0, "intercept", 1.0)
    y = df["accept"].to_numpy(dtype=float)
    xm = x.to_numpy(dtype=float)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except TypeError:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
    model.fit(xm, y)
    p = model.predict_proba(xm)[:, 1]
    w = p * (1 - p)
    bread = np.linalg.pinv(xm.T @ (xm * w[:, None]))
    scores = xm * (y - p)[:, None]
    grouped = pd.DataFrame(scores, columns=x.columns).groupby(df["task_id"].to_numpy()).sum().to_numpy()
    meat = grouped.T @ grouped
    cov = bread @ meat @ bread
    n, k = xm.shape
    g = len(np.unique(df["task_id"]))
    cov *= (g / (g - 1)) * ((n - 1) / (n - k))
    j = list(x.columns).index("same")
    beta = float(model.coef_[0][j])
    se = float(np.sqrt(cov[j, j]))
    z = beta / se
    result = {"beta_same_family": beta, "odds_ratio": float(np.exp(beta)),
              "clustered_se": se, "z": z, "p_two_sided": float(2 * norm.sf(abs(z))),
              "ci95": [beta - 1.959964 * se, beta + 1.959964 * se],
              "n": int(n), "tasks": int(g), "features": list(x.columns)}
    if include_provider:
        jp = list(x.columns).index("same_provider")
        bp = float(model.coef_[0][jp])
        sp = float(np.sqrt(cov[jp, jp]))
        result.update({"beta_same_provider": bp,
                       "p_same_provider": float(2 * norm.sf(abs(bp / sp)))})
    return result


def far_matrix(df: pd.DataFrame) -> dict:
    out = {}
    for e in MODELS:
        out[NAMES[e]] = {}
        for g in MODELS:
            cell = df[(df.evaluator == e) & (df.generator == g)]
            out[NAMES[e]][NAMES[g]] = {
                "far": 100 * cell.accept.mean(), "n": int(len(cell))}
    return out


PROD = {}
for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l)
        PROD[(r["model"], r["task_id"])] = r

results = {}
lines = ["Crossed evaluator x generator analysis", "=====================================",
         "Incorrect candidates only. Logistic acceptance model: evaluator and generator",
         "categorical effects + same-family indicator; task-clustered sandwich SEs.", ""]
for arm in ["selfcheck", "steelman"]:
    df = load_records(arm)
    fit = fit_clustered(df)
    matrix = far_matrix(df)
    results[arm] = {"fit": fit, "matrix": matrix}
    provider_fit = fit_clustered(df, include_provider=True)
    results[arm]["provider_sensitivity"] = provider_fit
    lines += [f"--- {arm} ---",
              f"n={fit['n']}, tasks={fit['tasks']}",
              f"same-family beta={fit['beta_same_family']:.4f}, OR={fit['odds_ratio']:.3f}, "
              f"clustered SE={fit['clustered_se']:.4f}, z={fit['z']:.2f}, "
              f"p={fit['p_two_sided']:.4g}, 95% CI [{fit['ci95'][0]:.4f},{fit['ci95'][1]:.4f}]",
              f"with same-provider indicator: same-family beta={provider_fit['beta_same_family']:.4f}, "
              f"OR={provider_fit['odds_ratio']:.3f}, p={provider_fit['p_two_sided']:.4g}; "
              f"same-provider beta={provider_fit['beta_same_provider']:.4f}, "
              f"p={provider_fit['p_same_provider']:.4g}",
              "FAR % matrix (rows evaluator, columns generator):"]
    lines.append("evaluator".ljust(12) + "".join(NAMES[g].rjust(12) for g in MODELS))
    for e in MODELS:
        lines.append(NAMES[e].ljust(12) + "".join(
            f"{matrix[NAMES[e]][NAMES[g]]['far']:12.1f}" for g in MODELS))
    lines.append("")

out = ROOT / "data/mbpp_v2/cross_eval"
(out / "crossed_model_summary.txt").write_text("\n".join(lines))
(out / "crossed_model_results.json").write_text(json.dumps(results, indent=1))
print("\n".join(lines))
