import numpy as np
import pandas as pd
from typing import Optional, Dict, Iterable
from scipy.stats import ks_2samp, chisquare
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def compare_distributions(
    before: pd.DataFrame,
    after: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    feature_types: Optional[Dict[str, str]] = None,   # {"feat": "numeric"|"categorical"}
    bins: int = 20,
    min_count_for_cat: int = 1,
    plot: bool = False,
    fig_cols: int = 2,
    fig_height: float = 3.0,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """
    Compare feature distributions before vs after simulation.

    Metrics
    -------
    numeric: PSI, JS (Jensen–Shannon), KS
    categorical: PSI, JS, chi2_p

    Parameters
    ----------
    before, after : pd.DataFrame
        DataFrames with the same feature columns.
    features : list[str] or None
        Which columns to compare (default: intersection of columns).
    feature_types : dict[str, "numeric"|"categorical"] or None
        Optional explicit typing; otherwise inferred (object/category -> categorical).
    bins : int
        Number of quantile bins for numeric PSI/JS.
    min_count_for_cat : int
        Minimum occurrences to keep a category; rarer ones collapse to "OTHER".
    plot : bool
        If True, show matplotlib plots for each feature.
    fig_cols : int
        Number of subplot columns if plot=True.
    fig_height : float
        Height per subplot row if plot=True.
    epsilon : float
        Smoothing to avoid zero division/log.

    Returns
    -------
    pd.DataFrame with columns:
        ['feature','type','psi','js','ks','chi2_p','n_before','n_after']
        (ks is NaN for categoricals; chi2_p is NaN for numerics)
    """
    # --- helpers ---
    def _infer_type(s: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        if pd.api.types.is_bool_dtype(s):
            return "categorical"
        if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
            return "categorical"
        return "numeric"  # fallback

    def _psi(p: np.ndarray, q: np.ndarray) -> float:
        # Population Stability Index: sum((q-p)*log(q/p))
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return float(np.sum((q - p) * np.log(q / p)))

    def _js(p: np.ndarray, q: np.ndarray) -> float:
        # Jensen–Shannon distance; convert to divergence (square) if preferred
        p = np.clip(p, epsilon, 1); p = p / p.sum()
        q = np.clip(q, epsilon, 1); q = q / q.sum()
        return float(jensenshannon(p, q, base=2))  # distance in [0,1]

    def _numeric_bins(x: pd.Series) -> np.ndarray:
        # robust quantile bins from BEFORE (to avoid leakage)
        # ensure strictly increasing edges; fall back to uniform if needed
        vals = x.dropna().values
        if len(vals) == 0:
            return np.array([-np.inf, np.inf])
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(vals, qs)
        edges = np.unique(edges)
        if len(edges) < 3:  # too many ties
            lo, hi = np.min(vals), np.max(vals)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, min(bins, 10) + 1)
        # pad to cover all
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _hist_counts(x: pd.Series, edges: np.ndarray) -> np.ndarray:
        # includes NaN bin
        mask_nan = x.isna().values
        counts, _ = np.histogram(x[~mask_nan].values, bins=edges)
        counts = counts.astype(float)
        # append NaN as last bucket
        return np.concatenate([counts, [mask_nan.sum()]])

    def _cat_counts(x: pd.Series) -> (np.ndarray, np.ndarray):
        s = x.astype("object").copy()
        # collapse rare categories (based on BEFORE only)
        vc = s.value_counts(dropna=False)
        keep = set(vc[vc >= min_count_for_cat].index)
        s.loc[~s.isin(keep)] = "OTHER"
        # ensure NaN token
        s = s.fillna("NaN")
        cats = np.array(sorted(s.unique()), dtype=object)
        cnt = s.value_counts().reindex(cats).fillna(0).values.astype(float)
        return cnt, cats

    def _align_cat_counts(bcnt: np.ndarray, bcat: np.ndarray,
                          acnt: np.ndarray, acat: np.ndarray):
        cats = sorted(set(bcat).union(set(acat)))
        bmap = dict(zip(bcat, bcnt)); amap = dict(zip(acat, acnt))
        b = np.array([bmap.get(c, 0.0) for c in cats], dtype=float)
        a = np.array([amap.get(c, 0.0) for c in cats], dtype=float)
        return b, a, np.array(cats, dtype=object)

    # --- setup ---
    if features is None:
        features = [c for c in before.columns if c in after.columns]
    if feature_types is None:
        feature_types = {}
    results = []

    # Plot layout preparation
    if plot:
        n = len(features)
        rows = int(np.ceil(n / fig_cols))
        fig, axes = plt.subplots(rows, fig_cols, figsize=(6 * fig_cols, fig_height * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

    for idx, feat in enumerate(features):
        s0 = before[feat]
        s1 = after[feat]
        ftype = feature_types.get(feat, _infer_type(s0))

        n0 = s0.shape[0] - s0.isna().sum()
        n1 = s1.shape[0] - s1.isna().sum()

        if ftype == "numeric":
            edges = _numeric_bins(s0)
            # counts + NaN bucket
            c0 = _hist_counts(s0, edges)
            c1 = _hist_counts(s1, edges)
            p0 = c0 / max(c0.sum(), 1.0)
            p1 = c1 / max(c1.sum(), 1.0)
            psi_val = _psi(p0, p1)
            js_val = _js(p0, p1)
            # KS on finite values only
            ks_val = np.nan
            try:
                ks_val = float(ks_2samp(s0.dropna().values, s1.dropna().values).statistic)
            except Exception:
                ks_val = np.nan

            results.append({
                "feature": feat, "type": "numeric",
                "psi": psi_val, "js": js_val, "ks": ks_val, "chi2_p": np.nan,
                "n_before": int(n0), "n_after": int(n1)
            })

            if plot:
                ax = axes[idx]
                # histogram overlays
                finite0 = s0[np.isfinite(s0)]
                finite1 = s1[np.isfinite(s1)]
                # Use aligned bin edges (excluding the +NaN bucket)
                edges_plot = edges.copy()
                edges_plot[0] = np.isfinite(finite0).any() and np.min(finite0) or -1
                edges_plot[-1] = np.isfinite(finite0).any() and np.max(finite0) or 1
                ax.hist(finite0, bins=edges, density=True, alpha=0.5, label="before")
                ax.hist(finite1, bins=edges, density=True, alpha=0.5, label="after")
                ax.set_title(f"{feat} | PSI={psi_val:.3f} JS={js_val:.3f} KS={ks_val:.3f}")
                ax.legend()
                ax.set_xlabel(feat); ax.set_ylabel("density")

        else:  # categorical
            bcnt, bcat = _cat_counts(s0)
            acnt, acat = _cat_counts(s1)
            bcnt, acnt, cats = _align_cat_counts(bcnt, bcat, acnt, acat)

            p0 = bcnt / max(bcnt.sum(), 1.0)
            p1 = acnt / max(acnt.sum(), 1.0)
            psi_val = _psi(p0, p1)
            js_val = _js(p0, p1)

            # Chi-square goodness-of-fit: expected from before, observed from after
            # Add smoothing to avoid zero expected
            expected = np.clip(p0, epsilon, None) * max(acnt.sum(), 1.0)
            chi2_stat, chi2_p = chisquare(f_obs=acnt, f_exp=expected)

            results.append({
                "feature": feat, "type": "categorical",
                "psi": psi_val, "js": js_val, "ks": np.nan, "chi2_p": float(chi2_p),
                "n_before": int(n0), "n_after": int(n1)
            })

            if plot:
                ax = axes[idx]
                x = np.arange(len(cats))
                width = 0.4
                ax.bar(x - width/2, p0, width=width, label="before")
                ax.bar(x + width/2, p1, width=width, label="after")
                ax.set_xticks(x)
                ax.set_xticklabels([str(c)[:18] for c in cats], rotation=30, ha="right")
                ax.set_title(f"{feat} | PSI={psi_val:.3f} JS={js_val:.3f} χ²p={chi2_p:.3f}")
                ax.legend()
                ax.set_ylabel("proportion")

    if plot:
        # hide any unused axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)
