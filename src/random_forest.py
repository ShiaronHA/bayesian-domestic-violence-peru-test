from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


def feature_selection(train, X_val, target_col):
    X = train.drop(target_col, axis=1)
    y = train[target_col]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    selected_features = X.columns[importances > 0.01]  # threshold tunable

    print("Features selected by RandomForest:")
    print(selected_features)

    X_val_aligned = X_val.reindex(columns=X.columns, fill_value=0)
    X_train_selected = X[selected_features]
    X_val_selected = X_val_aligned[selected_features]

    return X_train_selected, X_val_selected


def learn_with_random_forest(train, target_col, val, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]

    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

    X_train, X_val = feature_selection(train, X_val, target_col)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)

    train_metrics = {
        'model': 'RandomForest',
        'dataset': 'train',
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }

    val_metrics = {
        'model': 'RandomForest',
        'dataset': 'val',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    }

    for metrics in [train_metrics, val_metrics]:
        print(f"\nMetrics ({metrics['dataset']}):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")

    metrics_df = pd.DataFrame([train_metrics, val_metrics])
    metrics_file_path = os.path.join(output_dir, 'metrics_random_forest.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMetrics saved to: {metrics_file_path}")

    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    conf_matrix_file_path = os.path.join('./plots', 'confusion_matrix_rf.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_file_path}")

    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, 'classification_report_rf.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to: {report_path}")

    return rf_model


def main():
    # 1. Load training and validation DataFrames
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    print("DataFrames cargados correctamente.")
    
    train_encoded.dropna(inplace=True)
    val_encoded.dropna(inplace=True)
    learn_with_random_forest(train_encoded, 'NIVEL_DE_RIESGO_VICTIMA', val_encoded, './results')
    print("Random Forest model trained successfully.")


if __name__ == "__main__":
    main()
