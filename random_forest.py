from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer


def feature_selection(train, X_val, target_col):
    X = train.drop(target_col, axis=1)
    y = train[target_col]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    selected_features = X.columns[importances > 0.01]  # Ajusta el umbral si es necesario

    print("Características seleccionadas por RandomForest:")
    print(selected_features)

    X_val_aligned = X_val.reindex(columns=X.columns, fill_value=0)
    X_train_selected = X[selected_features]
    X_val_selected = X_val_aligned[selected_features]

    return X_train_selected, X_val_selected


def learn_with_random_forest(train, target_col, val, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('./plots', exist_ok=True)

    # Separar características y etiquetas
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]

    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Selección de características usando RandomForest
    X_train, X_val = feature_selection(train, X_val, target_col)

    print(f"Forma de X_train_lasso: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val_lasso: {X_val.shape}, y_val: {y_val.shape}")

    # Entrenar modelo
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)

    # Métricas para train
    train_metrics = {
        'model': 'RandomForest',
        'dataset': 'train',
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    }

    # Métricas para val
    val_metrics = {
        'model': 'RandomForest',
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
    metrics_file_path = os.path.join(output_dir, 'metrics_random_forest.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMétricas guardadas en: {metrics_file_path}")

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Validación')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    conf_matrix_file_path = os.path.join('./plots', 'confusion_matrix_rf.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {conf_matrix_file_path}")

    # Guardar reporte de clasificación como texto
    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, 'classification_report_rf.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Reporte de clasificación guardado en: {report_path}")

    return rf_model


def main():
    # 1. Leemos los DataFrames de entrenamiento y validación
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    print("DataFrames cargados correctamente.")

    train_encoded.info()

    # 2. Entrenamos el modelo Random Forest con los datos de entrenamiento
    model_rf = learn_with_random_forest(train_encoded, 'NIVEL_DE_RIESGO_VICTIMA', val_encoded, './results')
    print("Modelo Random Forest entrenado correctamente.")

    # Evaluación del modelo aprendido
    print("\nEvaluando el modelo aprendido...")

    # Guardar modelo entrenado
    # model_dir = './models'
    # os.makedirs(model_dir, exist_ok=True)
    # model_path1 = os.path.join(model_dir, 'random_forest_model_part1.pkl')
    # model_path2 = os.path.join(model_dir, 'random_forest_model_part2.pkl')

    # # Serializar en memoria
    # model_bytes = pickle.dumps(model_rf)

    # # Dividir en 2 partes
    # split_index = len(model_bytes) // 2
    # with open(model_path1, 'wb') as f1:
    #     f1.write(model_bytes[:split_index])
    # with open(model_path2, 'wb') as f2:
    #     f2.write(model_bytes[split_index:])

    # print(f"Modelo guardado en dos partes:\n - {model_path1}\n - {model_path2}")

    # (Opcional: descomentar si se desea verificar la carga del modelo)
    # with open(model_path1, 'rb') as f1, open(model_path2, 'rb') as f2:
    #     model_bytes = f1.read() + f2.read()
    # model_rf = pickle.loads(model_bytes)
    # print("Modelo cargado correctamente desde dos partes.")


if __name__ == "__main__":
    main()
