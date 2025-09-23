"""
GRI-Minimal: Guided vs Random label buying with fitted models

You provide:
  - A **fitted propensity pipeline** (predict_proba -> P(Accept=1)).
  - A **fitted credit risk pipeline** (predict_proba -> PD) trained on accepts-only.
  - A **label_oracle(indices)** function that returns true labels for the selected reject indices
    (in research, this may look up ground-truth; in production it would call a bureau API).

We then:
  1) Score propensity on train to find the *overlap* region among rejects.
  2) Use the fitted risk model to compute PD on rejects.
  3) Select a batch of rejects to label using:
      - GUIDED: uncertainty + proximity to current decision threshold (within overlap)
      - RANDOM: uniform random within overlap
  4) Acquire true labels for each batch and **retrain** fresh risk models on accepts + new labels.
  5) Evaluate each retrained model vs baseline at a **fixed approval rate** using Expected Loss (EL), ROC-AUC, PR-AUC.

Keep it simple. No DR here.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

# ------------------------- Metrics at fixed approval -------------------------

def fixed_approval_threshold(pd_scores: np.ndarray, approval_rate: float) -> float:
    assert 0 < approval_rate < 1, "approval_rate must be in (0,1)"
    return float(np.quantile(pd_scores, approval_rate))


def evaluate_at_approval_rate(
    pd_scores: np.ndarray,
    y_true: np.ndarray,
    ead: np.ndarray,
    lgd: np.ndarray,
    approval_rate: float,
) -> Dict[str, float]:
    pd_scores = np.asarray(pd_scores).ravel()
    y_true = np.asarray(y_true).ravel()
    ead = np.asarray(ead).ravel()
    lgd = np.asarray(lgd).ravel()

    thr = fixed_approval_threshold(pd_scores, approval_rate)
    approve = pd_scores <= thr
    el = float(np.sum(pd_scores[approve] * ead[approve] * lgd[approve]))

    out = {
        "threshold_pd": thr,
        "approval_rate": float(np.mean(approve)),
        "EL": el,
    }
    # Global ranking metrics (informational)
    try:
        out["roc_auc"] = roc_auc_score(y_true, -pd_scores)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc_bad"] = average_precision_score(y_true, -pd_scores)
    except Exception:
        out["pr_auc_bad"] = np.nan
    try:
        out["brier"] = brier_score_loss(y_true, pd_scores)
    except Exception:
        out["brier"] = np.nan
    return out

# ----------------------------- Selection utils ------------------------------

def propensity_overlap(pi: np.ndarray, eps: float = 0.05) -> np.ndarray:
    pi = np.asarray(pi).ravel()
    return (pi >= eps) & (pi <= 1 - eps)


def _z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + eps)


def uncertainty(pd_scores: np.ndarray, method: str = "entropy") -> np.ndarray:
    p = np.clip(np.asarray(pd_scores).ravel(), 1e-6, 1 - 1e-6)
    if method == "entropy":
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    elif method == "margin":
        return -np.abs(p - 0.5)
    else:
        raise ValueError("method must be 'entropy' or 'margin'")


def near_threshold(pd_scores: np.ndarray, approval_rate: float) -> np.ndarray:
    thr = fixed_approval_threshold(pd_scores, approval_rate)
    return -np.abs(pd_scores - thr)


def guided_select(
    pd_scores_reject: np.ndarray,
    pi_reject: np.ndarray,
    budget: int,
    approval_rate: float,
    eps: float = 0.05,
    alpha_uncert: float = 0.6,
    alpha_boundary: float = 0.4,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Return relative indices (within rejects) to buy under GUIDED policy."""
    pd_scores_reject = np.asarray(pd_scores_reject).ravel()
    pi_reject = np.asarray(pi_reject).ravel()

    mask = propensity_overlap(pi_reject, eps)
    elig = np.where(mask)[0]
    if elig.size == 0:
        return np.array([], dtype=int)

    u = uncertainty(pd_scores_reject[elig], method="entropy")
    b = near_threshold(pd_scores_reject[elig], approval_rate)
    s = alpha_uncert * _z(u) + alpha_boundary * _z(b)
    order = np.argsort(-s)
    return elig[order[: min(budget, elig.size)]]


def random_select(
    pi_reject: np.ndarray,
    budget: int,
    eps: float = 0.05,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    pi_reject = np.asarray(pi_reject).ravel()
    mask = propensity_overlap(pi_reject, eps)
    elig = np.where(mask)[0]
    if elig.size == 0:
        return np.array([], dtype=int)
    return rng.choice(elig, size=min(budget, elig.size), replace=False)

# ----------------------------- Main experiment ------------------------------

def run_minimal(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    A_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ead_test: np.ndarray,
    lgd_test: np.ndarray,
    propensity_pipe_fitted: Pipeline,
    risk_pipe_fitted: Pipeline,
    label_oracle: Callable[[np.ndarray], np.ndarray],
    budget: int = 1000,
    approval_rate: float = 0.5,
    eps: float = 0.05,
    risk_factory: Optional[Callable[[], ClassifierMixin]] = None,
    random_state: Optional[int] = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compare GUIDED vs RANDOM label acquisition using your fitted models.

    - `propensity_pipe_fitted` is used as-is to score pi on train.
    - `risk_pipe_fitted` is used as the baseline and to score PD on rejects for selection.
    - Retraining on augmented data uses `risk_factory()` if provided; else clones the estimator
      inside `risk_pipe_fitted` (last step must be an sklearn estimator) and rebuilds a pipeline
      with the same transformers (except the final estimator is fresh).
    """
    rng = np.random.RandomState(random_state)

    # 1) Propensity and masks
    pi = propensity_pipe_fitted.predict_proba(X_train)[:, 1]
    acc = A_train.astype(bool)
    rej = ~acc

    # 2) Baseline PD model for selection
    pd_rej = risk_pipe_fitted.predict_proba(X_train.iloc[rej])[:, 1]
    pi_rej = pi[rej]
    idx_rej_all = np.where(rej)[0]

    # 3) Two selection policies (relative indices within rejects)
    rel_guided = guided_select(pd_rej, pi_rej, budget, approval_rate, eps, random_state=random_state)
    rel_random = random_select(pi_rej, budget, eps, random_state=random_state)
    idx_guided = idx_rej_all[rel_guided]
    idx_random = idx_rej_all[rel_random]

    # 4) Unfold true labels via oracle (bureau)
    y_guided = label_oracle(idx_guided)
    y_random = label_oracle(idx_random)

    # 5) Helper: rebuild a fresh risk pipeline (keep pre-proc, swap estimator)
    def _fresh_risk_pipe_from_fitted(fitted: Pipeline) -> Pipeline:
        if risk_factory is None:
            # try to clone the last step estimator
            steps = fitted.steps
            preproc = steps[:-1]
            est_name, est_obj = steps[-1]
            fresh_est = clone(est_obj)
        else:
            steps = fitted.steps
            preproc = steps[:-1]
            fresh_est = risk_factory()
            est_name = steps[-1][0]
        new_steps = preproc + [(est_name, fresh_est)]
        return Pipeline(new_steps)

    # 6) Data for augmentation
    X_acc, y_acc = X_train.iloc[acc], y_train[acc]

    def _augment_and_fit(add_idx: np.ndarray, add_y: np.ndarray) -> Pipeline:
        pipe = _fresh_risk_pipe_from_fitted(risk_pipe_fitted)
        X_add = X_train.iloc[add_idx]
        y_add = add_y
        X_aug = pd.concat([X_acc, X_add], axis=0)
        y_aug = np.concatenate([y_acc, y_add])
        pipe.fit(X_aug, y_aug)
        return pipe

    pipe_guided = _augment_and_fit(idx_guided, y_guided)
    pipe_random = _augment_and_fit(idx_random, y_random)

    # 7) Evaluate baseline and both arms
    def _eval(pipe: Pipeline) -> Dict[str, float]:
        pd_test = pipe.predict_proba(X_test)[:, 1]
        return evaluate_at_approval_rate(pd_test, y_test, ead_test, lgd_test, approval_rate)

    metrics = {
        "baseline": _eval(risk_pipe_fitted),
        "guided": _eval(pipe_guided),
        "random": _eval(pipe_random),
    }

    # add deltas vs baseline
    for k in ("guided", "random"):
        metrics[k]["delta_EL_vs_baseline"] = metrics["baseline"]["EL"] - metrics[k]["EL"]
        if not np.isnan(metrics[k].get("roc_auc", np.nan)) and not np.isnan(metrics["baseline"].get("roc_auc", np.nan)):
            metrics[k]["delta_roc_auc_vs_baseline"] = metrics[k]["roc_auc"] - metrics["baseline"]["roc_auc"]
        if not np.isnan(metrics[k].get("pr_auc_bad", np.nan)) and not np.isnan(metrics["baseline"].get("pr_auc_bad", np.nan)):
            metrics[k]["delta_pr_auc_vs_baseline"] = metrics[k]["pr_auc_bad"] - metrics["baseline"]["pr_auc_bad"]

    # selection diagnostics
    metrics["diagnostics"] = {
        "n_rejects_total": int(np.sum(rej)),
        "n_guided": int(len(idx_guided)),
        "n_random": int(len(idx_random)),
        "overlap_eps": float(eps),
        "approval_rate_eval": float(approval_rate),
    }

    return metrics

# End of file
