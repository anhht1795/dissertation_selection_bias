import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal, Dict, Any, List, Callable, Union

def f_gini(y_actual, y_pred):
    """Calculate the Gini coefficient."""
    assert y_actual.shape == y_pred.shape, "Shapes of actual and predicted do not match."
    # calculate roc_auc
    auc = roc_auc_score(y_actual, y_pred)
    return 2*auc - 1

def f_roc_auc(y_actual, y_pred):
    """Calculate the Gini coefficient."""
    assert y_actual.shape == y_pred.shape, "Shapes of actual and predicted do not match."
    # calculate roc_auc
    auc = roc_auc_score(y_actual, y_pred)
    return auc

def f_pr_auc(y_actual, y_pred):
    """Calculate the PR AUC."""
    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)
    pr_auc = np.trapz(recall, precision)
    return pr_auc

def f_plot_auc(y_actual, y_pred):
    """Plot the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def f_plot_pr_auc(y_actual, y_pred):
    """Plot the Precision-Recall curve."""

    precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def f_ks(y_actual, y_pred):
    """Calculate the KS statistic."""
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    ks = max(tpr - fpr)
    return ks

def f_precision(y_actual, y_pred, threshold=0.5):
    """Calculate precision."""
    y_pred_label = (y_pred >= threshold).astype(int)
    return precision_score(y_actual, y_pred_label)

def f_recall(y_actual, y_pred, threshold=0.5):
    """Calculate recall."""
    y_pred_label = (y_pred >= threshold).astype(int)
    return recall_score(y_actual, y_pred_label)

def f_accuracy(y_actual, y_pred, threshold=0.5):
    """Calculate accuracy."""
    y_pred_label = (y_pred >= threshold).astype(int)
    return accuracy_score(y_actual, y_pred_label)

def f_fbeta_score(y_actual, y_pred, beta=1, threshold=0.5):
    """Calculate F-beta score."""
    y_pred_label = (y_pred >= threshold).astype(int)
    return f1_score(y_actual, y_pred_label, beta=beta)


def cutoff_confusion_matrix(y_true, y_score, cutoff_fraction=0.5, profit_margin=1.0, loss_cost=5.0):
    """
    Evaluate model performance at a given cutoff fraction of applicants.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels (1 = bad/default, 0 = good).
    y_score : array-like
        Predicted scores or probabilities (higher = riskier).
    cutoff_fraction : float, default=0.5
        Fraction of applicants to reject (e.g., 0.5 = reject 50% highest-risk).
    profit_margin : float, default=1.0
        Profit per good loan (proxy).
    loss_cost : float, default=5.0
        Cost per bad loan (proxy).
    
    Returns
    -------
    results : dict
        Confusion matrix components + key business metrics.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_score)
    cutoff_index = int((1 - cutoff_fraction) * n)  # accept bottom fraction
    threshold = np.sort(y_score)[cutoff_index]

    # Predictions: accept (0) if score below threshold, reject (1) if above
    y_pred = (y_score >= threshold).astype(int)  # 1 = reject, 0 = accept

    # Map to confusion matrix: TN, FP, FN, TP
    # In credit terms:
    # TN = bad correctly rejected
    # FP = good wrongly rejected
    # FN = bad wrongly accepted
    # TP = good correctly accepted
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Business metrics
    approval_rate = (tp + fn) / n
    bad_rate_accepted = fn / (tp + fn) if (tp + fn) > 0 else 0
    rejection_efficiency = tn / (tn + fp) if (tn + fp) > 0 else 0
    expected_profit = tp * profit_margin - fn * loss_cost

    results = {
        "cutoff_threshold": threshold,
        "TN (bad_rejected)": tn,
        "FP (good_rejected)": fp,
        "FN (bad_accepted)": fn,
        "TP (good_accepted)": tp,
        "approval_rate": approval_rate,
        "bad_rate_in_accepted": bad_rate_accepted,
        "rejection_efficiency": rejection_efficiency,
        "expected_profit": expected_profit
    }

    return results


# ===== Helper diagnostics for IPW =====
def standardized_mean_diff(X_a, X_b, w_a=None, w_b=None):
    """
    SMD by column: (mean_a - mean_b)/pooled_sd, supports weights.
    For categorical (0/1 dummies) works fine; for raw categoricals, use dummies first.
    Returns a pd.Series indexed by feature name.
    """
    def wmean(x, w):
        return np.average(x, weights=w) if w is not None else x.mean()
    def wvar(x, w):
        if w is None:
            return x.var(ddof=1)
        w = np.asarray(w)
        x = np.asarray(x)
        wm = np.average(x, weights=w)
        return np.average((x - wm)**2, weights=w)
    smds = {}
    for col in X_a.columns:
        ma = wmean(X_a[col].values, w_a)
        mb = wmean(X_b[col].values, w_b)
        va = wvar(X_a[col].values, w_a)
        vb = wvar(X_b[col].values, w_b)
        pooled_sd = np.sqrt(0.5*(va + vb) + 1e-12)
        smds[col] = (ma - mb)/pooled_sd
    return pd.Series(smds)

def effective_sample_size(weights):
    w = np.asarray(weights)
    return (w.sum()**2) / (np.sum(w**2) + 1e-12)

def weighted_ks_score(y_true, y_pred, sample_weight=None):
    # Weighted KS for binary labels
    df = pd.DataFrame({"y": y_true, "p": y_pred, "w": sample_weight if sample_weight is not None else 1.0})
    df = df.sort_values("p")
    w = df["w"].to_numpy()
    y = df["y"].to_numpy()
    w1 = np.where(y==1, w, 0)
    w0 = np.where(y==0, w, 0)
    cdf1 = np.cumsum(w1)/w1.sum() if w1.sum() > 0 else np.zeros_like(w1, dtype=float)
    cdf0 = np.cumsum(w0)/w0.sum() if w0.sum() > 0 else np.zeros_like(w0, dtype=float)
    return np.max(np.abs(cdf1 - cdf0)) if (w1.sum()>0 and w0.sum()>0) else np.nan

def weighted_brier(y_true, y_prob, sample_weight=None):
    err = (y_prob - y_true)**2
    if sample_weight is None:
        return err.mean()
    w = np.asarray(sample_weight)
    return np.average(err, weights=w)

def weighted_roc_auc(y_true, y_score, sample_weight=None):
    return roc_auc_score(y_true, y_score, sample_weight=sample_weight)

def weighted_prauc(y_true, y_score, sample_weight=None):
    # average_precision_score supports sample_weight
    return average_precision_score(y_true, y_score, sample_weight=sample_weight)

def tail_precision(
    p: np.ndarray,
    y: np.ndarray,
    mode: str = "percentile",          # "absolute" or "percentile"
    hi: float = 0.96,
    lo: float = 0.04,
    sample_weight: np.ndarray = None
):
    """
    Compute precision in the high and low probability tails.

    Parameters
    ----------
    p : array-like
        Predicted probabilities for y=1.
    y : array-like
        True binary labels {0,1}.
    mode : {"absolute","percentile"}
        - "absolute": use hi/lo as probability cutoffs (e.g., 0.96 / 0.04)
        - "percentile": use hi/lo as quantiles of p (e.g., 0.96 -> 96th percentile)
    hi, lo : float
        Thresholds interpreted per 'mode'.
    sample_weight : array-like or None
        Optional weights for precision calculation.

    Returns
    -------
    prec_pos, prec_neg, n_pos, n_neg, t_pos_abs, t_neg_abs
        Precision in the high tail, precision in the low tail,
        counts in each tail, and the absolute probability cutoffs used.
    """
    p = np.asarray(p)
    y = np.asarray(y)

    # Determine absolute thresholds
    if mode == "absolute":
        t_pos_abs, t_neg_abs = float(hi), float(lo)
    elif mode == "percentile":
        t_pos_abs = float(np.quantile(p, hi))
        t_neg_abs = float(np.quantile(p, lo))
    else:
        raise ValueError("mode must be 'absolute' or 'percentile'")

    pos_mask = (p >= t_pos_abs)
    neg_mask = (p <= t_neg_abs)

    def _wmean(mask, positive=True):
        n = int(mask.sum())
        if n == 0:
            return float("nan"), n
        if sample_weight is None:
            if positive:
                return float((y[mask] == 1).mean()), n
            else:
                return float((y[mask] == 0).mean()), n
        # weighted precision
        w = np.asarray(sample_weight)[mask]
        if positive:
            num = (w * (y[mask] == 1)).sum()
        else:
            num = (w * (y[mask] == 0)).sum()
        den = w.sum()
        return float(num / den) if den > 0 else float("nan"), n

    prec_pos, n_pos = _wmean(pos_mask, positive=True)
    prec_neg, n_neg = _wmean(neg_mask, positive=False)

    return prec_pos, prec_neg, n_pos, n_neg, t_pos_abs, t_neg_abs


def plot_probability_by_decile(
    y_prob: Union[np.ndarray, pd.Series],
    y_true: Optional[Union[np.ndarray, pd.Series]] = None,
    title: str = "Average Probability by Decile",
    figsize: Tuple[int, int] = (10, 6),
    color_scheme: str = "viridis",
    show_perfect_calibration: bool = True,
    show_counts: bool = False,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot average predicted probability and actual rate by decile.
    
    This function creates a decile analysis plot showing:
    - Average predicted probability per decile
    - Actual positive rate per decile (if y_true provided)
    - Perfect calibration line (optional)
    - Sample counts per decile (optional)
    
    Parameters:
    -----------
    y_prob : array-like  
        Predicted probabilities (between 0 and 1)
    y_true : array-like, optional
        True binary labels (0 or 1). If not provided, only predicted probabilities are shown
    title : str, default="Average Probability by Decile"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    color_scheme : str, default="viridis"
        Color scheme for the bars
    show_perfect_calibration : bool, default=True
        Whether to show the perfect calibration diagonal line (only relevant if y_true provided)
    show_counts : bool, default=True
        Whether to show sample counts on top of bars
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes  
        The axes object
    decile_stats : pandas.DataFrame
        DataFrame with decile statistics
        
    Example:
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> # Generate sample data
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> 
    >>> # Train model
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_prob = model.predict_proba(X_test)[:, 1]
    >>> 
    >>> # Plot decile analysis with true labels
    >>> fig, ax, stats = plot_probability_by_decile(y_prob, y_test)
    >>> plt.show()
    >>> 
    >>> # Plot only predicted probabilities
    >>> fig, ax, stats = plot_probability_by_decile(y_prob)
    >>> plt.show()
    """
    
    # Convert to numpy arrays
    y_prob = np.array(y_prob)
    if y_true is not None:
        y_true = np.array(y_true)
    
    # Validate inputs
    if y_true is not None and len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    
    if y_true is not None and not all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0s and 1s")
        
    if not all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be between 0 and 1")
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({'y_prob': y_prob})
    if y_true is not None:
        df['y_true'] = y_true
    
    # Create deciles based on predicted probability
    df['decile'] = pd.qcut(df['y_prob'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate statistics by decile
    agg_dict = {
        'y_prob': ['mean', 'min', 'max', 'count']
    }
    
    if y_true is not None:
        agg_dict['y_true'] = ['mean', 'sum']
    
    decile_stats = df.groupby('decile').agg(agg_dict).round(4)
    
    # Flatten column names
    if y_true is not None:
        decile_stats.columns = ['avg_pred_prob', 'min_pred_prob', 'max_pred_prob', 
                               'count', 'actual_rate', 'positive_count']
    else:
        decile_stats.columns = ['avg_pred_prob', 'min_pred_prob', 'max_pred_prob', 'count']
    
    decile_stats = decile_stats.reset_index()
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Set up colors
    colors = plt.cm.get_cmap(color_scheme)(np.linspace(0, 1, len(decile_stats)))
    
    # Create bar plot
    x_pos = np.arange(len(decile_stats))
    
    if y_true is not None:
        # Two bars when y_true is provided
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, decile_stats['avg_pred_prob'], width, 
                       label='Average Predicted Probability', color=colors, alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, decile_stats['actual_rate'], width,
                       label='Actual Positive Rate', color='lightcoral', alpha=0.8)
        bars = [bars1, bars2]
    else:
        # Single bar when only y_prob is provided
        bars1 = ax.bar(x_pos, decile_stats['avg_pred_prob'], 
                       label='Average Predicted Probability', color=colors, alpha=0.8)
        bars = [bars1]
    
    # Add perfect calibration line if requested and y_true is available
    if show_perfect_calibration and y_true is not None:
        ax.plot(x_pos, decile_stats['avg_pred_prob'], 'k--', alpha=0.7, 
                label='Perfect Calibration')
    
    # Add count annotations if requested
    if show_counts:
        for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, decile_stats['count'])):
            height = max(bar1.get_height(), bar2.get_height())
            ax.annotate(f'n={count}', 
                       xy=(i, height), 
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('Decile')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'D{i}' for i in decile_stats['decile']])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add text box with overall statistics
    overall_stats = f"""Overall Statistics:
    Samples: {len(df):,}
    Avg Pred Prob: {df['y_prob'].mean():.3f}"""
    
    ax.text(0.02, 0.98, overall_stats, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    plt.tight_layout()
    
    return fig, ax, decile_stats

def roc_gap_by_decile(
    y_true,
    y_proba,
    n_bins: int = 10,
    sample_weight=None,
    greater_is_riskier: bool = True,
) -> pd.DataFrame:
    """
    Compute (cumulative) TPR, FPR and their gap at decile thresholds of y_proba.
    This mirrors the KS-by-decile table often used in credit risk.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels (1 = bad/positive, 0 = good/negative).
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities or scores.
    n_bins : int
        Number of quantile bins (deciles -> 10).
    sample_weight : array-like or None
        Optional per-sample weights.
    greater_is_riskier : bool
        If True, higher scores mean higher risk (bad). If False, invert.

    Returns
    -------
    pd.DataFrame with one row per decile threshold (from low risk -> high risk):
        - decile: 1..n_bins (1 = lowest scores, n_bins = highest scores)
        - thr: the lower score bound of this decile
        - n, w: count and total weight in this decile
        - bad, good: weighted counts per decile
        - cum_bad, cum_good: cumulative (from this decile up to max risk)
        - TPR, FPR: cumulative rates at this threshold
        - GAP: TPR - FPR at this threshold
        - bin_bad_rate: within-bin bad rate (diagnostic)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if not greater_is_riskier:
        y_proba = -y_proba  # so that “larger is riskier” holds

    if sample_weight is None:
        w = np.ones_like(y_true, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)

    # Bin by quantiles (deciles). We guard against duplicate edges.
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y_proba, q))
    # If too few unique edges (pathological equal scores), fall back to rank-based bins
    if len(edges) <= 2:
        ranks = pd.Series(y_proba).rank(method="average") / len(y_proba)
        edges = np.unique(np.quantile(ranks, q))

    # Assign bins: 1..n_bins (1=lowest risk decile)
    # Use pandas.cut to get closed-left/open-right bins based on edges
    # Ensure we always produce exactly n_bins by clipping labels to 1..n_bins
    bins = pd.cut(y_proba, bins=edges, labels=False, include_lowest=True) + 1
    # If fewer unique bins due to ties at edges, coalesce to ≤ n_bins but proceed.
    df = pd.DataFrame({"decile": bins, "y": y_true, "score": y_proba, "w": w}).dropna()

    # Aggregate by decile
    g = df.groupby("decile", as_index=False).agg(
        n=("y", "size"),
        w=("w", "sum"),
        bad=("y", lambda s: np.dot(s, df.loc[s.index, "w"])),
    )
    g["good"] = g["w"] - g["bad"]

    # Lower bound (threshold) of each decile for reference
    thr_map = df.groupby("decile")["score"].min()
    g["thr"] = g["decile"].map(thr_map)

    # Sort by ascending score (decile 1 = lowest risk)
    g = g.sort_values(["thr", "decile"]).reset_index(drop=True)

    # Cumulative from this decile UPWARD (toward higher risk):
    # reverse cumsum then flip back
    g["cum_bad"] = g["bad"][::-1].cumsum()[::-1]
    g["cum_good"] = g["good"][::-1].cumsum()[::-1]

    total_bad = g["bad"].sum()
    total_good = g["good"].sum()
    # Avoid division by zero
    g["TPR"] = np.where(total_bad > 0, g["cum_bad"] / total_bad, 0.0)
    g["FPR"] = np.where(total_good > 0, g["cum_good"] / total_good, 0.0)
    g["GAP"] = g["TPR"] - g["FPR"]

    # Within-bin bad rate (diagnostic only; not KS)
    g["bin_bad_rate"] = np.where(g["w"] > 0, g["bad"] / g["w"], np.nan)

    # Final tidy columns
    out = g[
        ["decile", "thr", "n", "w", "bad", "good", "cum_bad", "cum_good", "TPR", "FPR", "GAP", "bin_bad_rate"]
    ].copy()
    # Ensure decile is int and contiguous starting at 1
    out["decile"] = out["decile"].astype(int)

    return out