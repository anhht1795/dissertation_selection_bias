import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score

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