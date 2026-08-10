# ============================================================
# EVALUATION MODULE: visualizations.py
# Generates Confusion Matrix and ROC Curve plots
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

class EvaluationVisualizer:
    """
    Generates and saves visual reports for AI system performance.
    """
    @staticmethod
    def plot_confusion_matrix(cm, labels, title="System Confusion Matrix"):
        """Plots a heatmap of the confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        
        plt.title(title, fontweight='bold', fontsize=14)
        plt.xlabel('Predicted Diagnosis', fontweight='bold')
        plt.ylabel('True Diagnosis', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = "system_confusion_matrix.png"
        plt.savefig(filename, dpi=150)
        print(f"✅ Saved visualization: {filename}")
        plt.show()

    @staticmethod
    def plot_roc_curve(y_true, y_prob, labels):
        """Plots the ROC Curve for multi-class classification"""
        # Binarize labels for One-vs-Rest ROC calculation
        y_bin = label_binarize(y_true, classes=labels)
        n_classes = len(labels)

        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each disease class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')

        # Plot baseline
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = "system_roc_curve.png"
        plt.savefig(filename, dpi=150)
        print(f"✅ Saved visualization: {filename}")
        plt.show()