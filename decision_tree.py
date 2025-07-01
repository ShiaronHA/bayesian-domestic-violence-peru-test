from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd


def feature_selection(train, X_val, target_col):
    X = train.drop(target_col, axis=1)
    y = train[target_col]

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)

    importances = dt.feature_importances_
    selected_features = X.columns[importances > 0.01]  # Ajusta umbral si es necesario

    print("Características seleccionadas por DecisionTree:")
    print(selected_features)

    X_val_aligned = X_val.reindex(columns=X.columns, fill_value=0)
    X_train_selected = X[selected_features]
    X_val_selected = X_val_aligned[selected_features]

    return X_train_selected, X_val_selected


def learn_with_decision_tree(train, target_col, val, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    # Separar características y etiquetas
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]

    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Selección de características usando DecisionTree
    X_train, X_val = feature_selection(train, X_val, target_col)

    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Entrenar modelo
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = dt_model.predict(X_train)
    y_val_pred = dt_model.predict(X_val)

    # Métricas para train
    train_metrics = {
        'model': 'DecisionTree',
        'dataset': 'train',
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }

    # Métricas para val
    val_metrics = {
        'model': 'DecisionTree',
        'dataset': 'val',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    }

    # Imprimir métricas
    for metrics in [train_metrics, val_metrics]:
        print(f"\nMétricas ({metrics['dataset']}):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")

    # Guardar métricas en CSV
    metrics_df = pd.DataFrame([train_metrics, val_metrics])
    metrics_file_path = os.path.join(output_dir, 'metrics_decision_tree.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMétricas guardadas en: {metrics_file_path}")

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Validación (Decision Tree)')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    conf_matrix_file_path = os.path.join('./plots', 'confusion_matrix_dt.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {conf_matrix_file_path}")

    # Reporte de clasificación
    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, 'classification_report_dt.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Reporte de clasificación guardado en: {report_path}")

    return dt_model


def main():
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    print("DataFrames cargados correctamente.")

    # Eliminar nulos
    train_encoded.dropna(inplace=True)
    val_encoded.dropna(inplace=True)

    # Entrenar modelo con árbol de decisión
    model_dt = learn_with_decision_tree(train_encoded, 'NIVEL_DE_RIESGO_VICTIMA', val_encoded, './results')
    print("Modelo Decision Tree entrenado correctamente.")


if __name__ == "__main__":
    main()
