import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix

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
