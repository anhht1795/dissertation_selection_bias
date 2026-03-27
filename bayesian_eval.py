"""
    # The module provides:
    # - Beta-binomial priors (global or per-score-bin) for rejected labels
    # - Monte Carlo estimation of metrics (ROC-AUC, PR-AUC, KS, Brier, calibration curve)
    # - Profit/acceptance-rate curves with uncertainty bands
    # - Utilities to bin scores, set priors, and summarize posterior estimates
    ----------------

    Bayesian evaluation for credit scoring under selection bias (accept-only labels).

    This module estimates portfolio-level model performance and business outcomes
    by integrating *known labels for accepts* with *Bayesian priors over rejects' labels*.
    It is intended for post-training evaluation of any scoring model, especially when
    rejected applicants' outcomes are missing (MAR/MNAR).

    Core ideas:
    - Treat unknown labels of rejected applicants as random variables.
    - Specify a prior over the default rate of rejects (global or per score bin).
    - Monte Carlo (MC) sample reject labels from the prior to obtain a posterior
    distribution over metrics: ROC-AUC, PR-AUC, KS, Brier, calibration, and profit
    at chosen acceptance rates.

    Design highlights:
    - **Priors**:
    - Global Beta(a, b) prior for reject default rate (simple, transparent).
    - Per-bin Beta priors (Dirichlet-Beta factorization) to reflect risk heterogeneity
        across score bands without assuming a calibration model.
    - **Metrics**:
    - ROC-AUC, PR-AUC, KS
    - Brier Score (overall)
    - Calibration curve (bins) for all applicants
    - Profit vs. acceptance-rate with uncertainty bands
    - **Acceptance policy**:
    - Threshold by **model score** to achieve a target acceptance rate over the
        full applicant pool (accepts + rejects).

    Dependencies: numpy, pandas, scikit-learn

    Example
    -------
    >>> import numpy as np
    >>> from bayesian_eval import (
    ...     BetaPriors, bin_scores, BayesianEvaluator, summarize_ci
    ... )
    >>> # Known accepts (labels), and scores for everyone
    >>> y_accepts = np.array([0,1,0,0,1, ...])
    >>> s_accepts = np.array([0.1,0.8,0.2, ...])
    >>> s_rejects = np.array([0.05,0.3,0.6, ...])
    >>>
    >>> # Set a simple global prior: rejects' PD ~ Beta(2, 8) (mean 0.2)
    >>> priors = BetaPriors.global_prior(a=2.0, b=8.0)
    >>>
    >>> # Or per-bin priors: 10 quantile bins with weakly-informative Beta(1,1)
    >>> bins = bin_scores(np.concatenate([s_accepts, s_rejects]), n_bins=10)
    >>> priors_bin = BetaPriors.per_bin_prior(n_bins=10, a=1.0, b=1.0)
    >>>
    >>> ev = BayesianEvaluator(
    ...     y_accepts=y_accepts,
    ...     s_accepts=s_accepts,
    ...     s_rejects=s_rejects,
    ...     bins=bins,
    ...     priors=priors_bin,     # or priors
    ...     random_state=42
    ... )
    >>>
    >>> # Evaluate at acceptance rates and get posterior summaries
    >>> acc_rates = [0.2, 0.3, 0.4, 0.5]
    >>> out = ev.evaluate(
    ...     n_mc=2000,
    ...     acceptance_rates=acc_rates,
    ...     profit_params=dict(
    ...         revenue_good=100.0,    # margin per good (non-default) account
    ...         loss_bad=300.0,        # loss per bad (default) account
    ...     ),
    ...     calibration_bins=10
    ... )
    >>> # Summarize AUC posterior
    >>> summarize_ci(out["auc_posterior"], q=(0.05, 0.5, 0.95))
    {'low': 0.721, 'med': 0.742, 'high': 0.762}
    >>> # Profit curve posterior summaries
    >>> out["profit_curve_summary"]
    # dict: acceptance_rate -> {'low': ..., 'med': ..., 'high': ...}

    from dataclasses import dataclass
    from typing import Optional, Dict, Tuple, List, Iterable
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.utils import check_random_state
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.utils import check_random_state


# --------------------------- Priors & Binning ---------------------------

@dataclass
class BetaPriors:
    """
    Container for Beta priors over reject default rates.

    Two modes:
    - Global prior:   use a single Beta(a,b) for all rejects
    - Per-bin priors: supply a vector of (a,b) per score bin

    Attributes
    ----------
    mode : {'global','per_bin'}
    a : float or np.ndarray
        For global: scalar alpha; for per_bin: vector of length n_bins
    b : float or np.ndarray
        For global: scalar beta; for per_bin: vector of length n_bins
    n_bins : Optional[int]
        Required for per_bin
    """
    mode: str
    a: np.ndarray
    b: np.ndarray
    n_bins: Optional[int] = None

    @staticmethod
    def global_prior(a: float, b: float) -> "BetaPriors":
        return BetaPriors(mode="global", a=np.array([a], dtype=float), b=np.array([b], dtype=float), n_bins=None)

    @staticmethod
    def per_bin_prior(n_bins: int, a: float = 1.0, b: float = 1.0) -> "BetaPriors":
        return BetaPriors(mode="per_bin", a=np.full(n_bins, a, dtype=float), b=np.full(n_bins, b, dtype=float), n_bins=n_bins)

    def set_bin_params(self, a: Iterable[float], b: Iterable[float]) -> "BetaPriors":
        """Set per-bin a,b vectors explicitly (length must equal n_bins)."""
        if self.mode != "per_bin":
            raise ValueError("set_bin_params is only valid for per_bin mode.")
        a = np.asarray(list(a), dtype=float)
        b = np.asarray(list(b), dtype=float)
        if len(a) != self.n_bins or len(b) != self.n_bins:
            raise ValueError("Length mismatch for per-bin priors.")
        self.a = a
        self.b = b
        return self


def bin_scores(scores: np.ndarray, n_bins: int = 10, strategy: str = "quantile") -> Dict[str, np.ndarray]:
    """
    Bin scores for thresholding and per-bin priors.

    Parameters
    ----------
    scores : array-like
        Scores for all applicants (accepts + rejects). Higher = riskier or safer, but must be consistent.
    n_bins : int
        Number of bins.
    strategy : {'quantile','uniform'}
        Quantile: equal count bins; uniform: equal width bins in [min,max].

    Returns
    -------
    dict with keys:
        'edges': np.ndarray of length n_bins+1
        'assign': np.ndarray of bin indices for each score (0..n_bins-1)
    """
    x = np.asarray(scores, dtype=float)
    if strategy == "quantile":
        # unique quantiles to avoid identical edges
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x, qs, method="nearest")
        # enforce strict monotonicity
        edges = np.unique(edges)
        if len(edges) - 1 < n_bins:
            # fall back to uniform bins if many ties
            edges = np.linspace(x.min(), x.max(), n_bins + 1)
    elif strategy == "uniform":
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    assign = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges)-2)
    return {"edges": edges, "assign": assign}


# --------------------------- Evaluator ---------------------------

@dataclass
class BayesianEvaluator:
    """
    Bayesian evaluator for selection-biased scoring models.

    Required inputs
    ---------------
    y_accepts : np.ndarray of shape (n_accepts,)
        Observed labels (0=good, 1=bad) for accepted applications.
    s_accepts : np.ndarray of shape (n_accepts,)
        Model scores for accepted applications.
    s_rejects : np.ndarray of shape (n_rejects,)
        Model scores for rejected applications (labels missing).
    bins : dict
        Output from bin_scores applied on ALL scores (accepts + rejects). Used for per-bin priors
        and acceptance thresholding by rate.
    priors : BetaPriors
        Beta priors for rejects.

    Notes
    -----
    - Score direction: larger score = larger default risk is assumed. If your score has the opposite
      direction, pass `invert_scores=True` to flip internally.
    """
    y_accepts: np.ndarray
    s_accepts: np.ndarray
    s_rejects: np.ndarray
    bins: Dict[str, np.ndarray]
    priors: BetaPriors
    invert_scores: bool = False
    random_state: Optional[int] = None

    def __post_init__(self):
        self.rng = check_random_state(self.random_state)
        self.y_accepts = np.asarray(self.y_accepts).astype(int)
        self.s_accepts = np.asarray(self.s_accepts, dtype=float).copy()
        self.s_rejects = np.asarray(self.s_rejects, dtype=float).copy()
        if self.invert_scores:
            self.s_accepts = -self.s_accepts
            self.s_rejects = -self.s_rejects
        # concat arrays for applicant-level computations
        self.s_all = np.concatenate([self.s_accepts, self.s_rejects])
        self.n_accepts = len(self.s_accepts)
        self.n_rejects = len(self.s_rejects)
        self.assign_all = self.bins["assign"]
        if len(self.assign_all) != len(self.s_all):
            raise ValueError("bins['assign'] length must equal len(s_accepts)+len(s_rejects).")

    # ---- sampling of reject labels ----

    def _sample_reject_labels(self) -> np.ndarray:
        """
        Sample rejected labels y_r ~ Bernoulli(p_r) under the specified prior.

        Returns
        -------
        y_r : np.ndarray of shape (n_rejects,), dtype=int {0,1}
        """
        if self.priors.mode == "global":
            a, b = float(self.priors.a[0]), float(self.priors.b[0])
            p = self.rng.beta(a, b, size=1)[0]
            y_r = (self.rng.random(self.n_rejects) < p).astype(int)
            return y_r
        elif self.priors.mode == "per_bin":
            if self.priors.n_bins is None:
                raise ValueError("per_bin prior requires n_bins")
            # get assign indices for rejects
            assign_rej = self.assign_all[self.n_accepts:]
            y_r = np.empty(self.n_rejects, dtype=int)
            for bidx in range(self.priors.n_bins):
                mask = (assign_rej == bidx)
                cnt = mask.sum()
                if cnt == 0:
                    continue
                a_b = float(self.priors.a[bidx])
                b_b = float(self.priors.b[bidx])
                p = self.rng.beta(a_b, b_b, size=1)[0]
                y_r[mask] = (self.rng.random(cnt) < p).astype(int)
            return y_r
        else:
            raise ValueError("Unknown prior mode.")

    # ---- acceptance thresholding by acceptance rate ----

    def _threshold_for_acceptance_rate(self, rate: float) -> float:
        """
        Compute score threshold such that the fraction of *accepted* applicants equals `rate`.
        Larger scores -> riskier; we *accept* those with scores below the threshold.

        Parameters
        ----------
        rate : float in (0,1]

        Returns
        -------
        thr : float
        """
        if not (0 < rate <= 1):
            raise ValueError("acceptance rate must be in (0,1].")
        # accept the lowest-risk fraction
        k = max(1, int(np.ceil(rate * len(self.s_all))))
        # np.partition selects k-th smallest
        thr = np.partition(self.s_all, k-1)[k-1]
        return thr

    # ---- metrics on one Monte Carlo draw ----

    def _metrics_one_draw(self, y_r: np.ndarray, acceptance_rates: Iterable[float], profit_params: Optional[Dict[str, float]], calibration_bins: int):
        """
        Compute metrics for a single Monte Carlo draw of reject labels.

        Returns a dict of scalars/arrays for this draw.
        """
        y_all = np.concatenate([self.y_accepts, y_r])
        s_all = self.s_all

        # ROC AUC (needs both classes)
        try:
            auc = roc_auc_score(y_all, s_all)
        except Exception:
            auc = np.nan

        # PR-AUC (average precision)
        try:
            pr_auc = average_precision_score(y_all, -s_all)  # non-default as "positive" if lower score=better
        except Exception:
            pr_auc = np.nan

        # KS
        try:
            fpr, tpr, _ = roc_curve(y_all, s_all)
            ks = np.max(np.abs(tpr - fpr))
        except Exception:
            ks = np.nan

        # Brier (overall)
        # Use a simple rank-to-prob mapping via isotonic-like bin mean (calibration curve bins)
        calib = calibration_curve(y_all, s_all, n_bins=calibration_bins)
        brier = np.mean((y_all - calib["p_hat_all"])**2)

        # Profit curve at acceptance rates
        profit_curve = {}
        if profit_params is not None:
            rev_good = float(profit_params.get("revenue_good", 0.0))
            loss_bad = float(profit_params.get("loss_bad", 0.0))
            for r in acceptance_rates:
                thr = self._threshold_for_acceptance_rate(r)
                accept_mask = s_all <= thr
                y_acc = y_all[accept_mask]
                n = len(y_acc)
                n_good = (y_acc == 0).sum()
                n_bad = n - n_good
                profit_curve[r] = rev_good * n_good - loss_bad * n_bad
        else:
            for r in acceptance_rates:
                profit_curve[r] = np.nan

        return {
            "auc": auc,
            "pr_auc": pr_auc,
            "ks": ks,
            "brier": brier,
            "profit_curve": profit_curve,
            "calibration": calib,
        }

    # ---- public API ----

    def evaluate(
        self,
        n_mc: int = 2000,
        acceptance_rates: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
        profit_params: Optional[Dict[str, float]] = None,
        calibration_bins: int = 20,
    ) -> Dict[str, object]:
        """
        Run Monte Carlo Bayesian evaluation and aggregate posterior summaries.

        Parameters
        ----------
        n_mc : int
            Monte Carlo samples.
        acceptance_rates : iterable of float
            Acceptance rates (0,1] at which to compute profit.
        profit_params : dict or None
            {'revenue_good': float, 'loss_bad': float}; if None, profit not computed.
        calibration_bins : int
            Bins for calibration curve.

        Returns
        -------
        dict with keys:
            - 'auc_posterior' : np.ndarray (n_mc,)
            - 'pr_auc_posterior' : np.ndarray (n_mc,)
            - 'ks_posterior' : np.ndarray (n_mc,)
            - 'brier_posterior' : np.ndarray (n_mc,)
            - 'profit_curve_posterior' : dict(rate -> np.ndarray (n_mc,))
            - 'profit_curve_summary' : dict(rate -> {'low','med','high'} at 5/50/95%)
            - 'calibration_last' : last-draw calibration dict (for plotting)
        """
        acceptance_rates = list(acceptance_rates)
        aucs = np.empty(n_mc)
        pras = np.empty(n_mc)
        kss = np.empty(n_mc)
        briers = np.empty(n_mc)
        profit_curves = {r: np.empty(n_mc) for r in acceptance_rates}
        calib_last = None

        for t in range(n_mc):
            y_r = self._sample_reject_labels()
            res = self._metrics_one_draw(y_r, acceptance_rates, profit_params, calibration_bins)
            aucs[t] = res["auc"]
            pras[t] = res["pr_auc"]
            kss[t] = res["ks"]
            briers[t] = res["brier"]
            for r in acceptance_rates:
                profit_curves[r][t] = res["profit_curve"][r]
            calib_last = res["calibration"]

        # Summaries for profit curve
        prof_summary = {r: summarize_ci(profit_curves[r]) for r in acceptance_rates}

        return {
            "auc_posterior": aucs,
            "pr_auc_posterior": pras,
            "ks_posterior": kss,
            "brier_posterior": briers,
            "profit_curve_posterior": profit_curves,
            "profit_curve_summary": prof_summary,
            "calibration_last": calib_last,
        }


# --------------------------- Calibration helper ---------------------------

def calibration_curve(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 20) -> Dict[str, np.ndarray]:
    """
    Simple calibration curve using score-based binning (by quantiles).

    Assumes higher score => higher risk (more likely y=1).

    Returns
    -------
    dict with keys:
      'bin_edges' : edges
      'bin_mean_score' : mean score per bin
      'bin_pos_rate' : empirical bad rate per bin
      'p_hat_all' : piecewise-constant mapping from score to estimated PD (array like y_true)
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    # quantile edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(s, qs, method="nearest")
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        edges = np.linspace(s.min(), s.max(), n_bins + 1)

    # assign bins
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, len(edges)-2)
    bin_mean = np.zeros(len(edges)-1)
    bin_pos = np.zeros(len(edges)-1)
    for b in range(len(edges)-1):
        mask = (idx == b)
        if mask.any():
            bin_mean[b] = s[mask].mean()
            bin_pos[b] = y[mask].mean()
        else:
            bin_mean[b] = 0.5*(edges[b]+edges[b+1])
            bin_pos[b] = y.mean()

    # map each point to its bin's pos rate
    p_hat_all = bin_pos[idx]

    return {
        "bin_edges": edges,
        "bin_mean_score": bin_mean,
        "bin_pos_rate": bin_pos,
        "p_hat_all": p_hat_all,
    }


# --------------------------- Summaries & Utilities ---------------------------

def summarize_ci(samples: np.ndarray, q: Tuple[float, float, float] = (0.05, 0.5, 0.95)) -> Dict[str, float]:
    """
    Summarize 1D posterior samples into (low, med, high) at chosen quantiles.
    """
    s = np.asarray(samples, dtype=float)
    lo, md, hi = np.quantile(s, q)
    return {"low": float(lo), "med": float(md), "high": float(hi)}


def ks_from_scores_labels(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Convenience KS computation.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(np.abs(tpr - fpr)))


def acceptance_threshold(scores_all: np.ndarray, rate: float) -> float:
    """
    Public helper: threshold for acceptance rate on all applicants.
    """
    scores_all = np.asarray(scores_all, dtype=float)
    k = max(1, int(np.ceil(rate * len(scores_all))))
    return np.partition(scores_all, k-1)[k-1]


__all__ = [
    "BetaPriors",
    "bin_scores",
    "BayesianEvaluator",
    "calibration_curve",
    "summarize_ci",
    "ks_from_scores_labels",
    "acceptance_threshold",
]
