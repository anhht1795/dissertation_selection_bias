import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt

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
    pr_auc = np.trapz(precision, recall)
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