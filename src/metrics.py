import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR  = os.path.join(_BASE_DIR, 'plots')
OUTPUT_DIR = os.path.join(_BASE_DIR, 'results')


def save_metrics(y_true, y_pred, label, model_name, output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR):
    """Computes, prints, and saves metrics, confusion matrix, and classification report.

    Args:
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        label:      File-name suffix used for all output files (e.g. 'random_forest').
        model_name: Model identifier stored in the metrics CSV.
        output_dir: Directory for CSV and TXT outputs.
        plots_dir:  Directory for confusion matrix PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    metrics = {
        'model':     model_name,
        'dataset':   'val',
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score':  f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    print(f"\nMetrics (val):")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")

    csv_path = os.path.join(output_dir, f'metrics_{label}.csv')
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    cm_path = os.path.join(plots_dir, f'confusion_matrix_{label}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    report_path = os.path.join(output_dir, f'classification_report_{label}.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_true, y_pred))
    print(f"Classification report saved to: {report_path}")
