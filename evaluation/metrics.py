# ============================================================
# EVALUATION MODULE: metrics.py
# Calculates Accuracy, Precision, Recall, F1-Score, ROC-AUC
# ============================================================

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score
)

class SystemEvaluator:
    """
    Computes performance metrics for the Healthcare Diagnostic AI.
    Handles multi-class classification evaluation.
    """
    def __init__(self, disease_labels):
        self.labels = disease_labels

    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculates all required capstone metrics comparing ground-truth 
        labels (y_true) against model predictions (y_pred).
        """
        # Calculate standard metrics using macro-averaging for multi-class
        metrics_report = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'Confusion Matrix': confusion_matrix(y_true, y_pred, labels=self.labels)
        }
        
        # Calculate ROC-AUC if probability scores are provided
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics_report['ROC-AUC'] = roc_auc
            except ValueError:
                metrics_report['ROC-AUC'] = "N/A (Requires probabilities for all classes)"
                
        return metrics_report

    def print_report(self, report):
        """Prints a cleanly formatted metrics report to the terminal"""
        print("\n" + "="*50)
        print("  🏥 SYSTEM EVALUATION METRICS REPORT")
        print("="*50)
        print(f"  Accuracy:  {report['Accuracy']:.4f}")
        print(f"  Precision: {report['Precision']:.4f}")
        print(f"  Recall:    {report['Recall']:.4f}")
        print(f"  F1-Score:  {report['F1-Score']:.4f}")
        if 'ROC-AUC' in report and isinstance(report['ROC-AUC'], float):
            print(f"  ROC-AUC:   {report['ROC-AUC']:.4f}")
        print("="*50 + "\n")