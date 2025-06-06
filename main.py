import pandas as pd
import pickle
import os
import json
import time
from sklearn.model_selection import train_test_split # Added import
from pgmpy.inference import BeliefPropagation
from pgmpy.readwrite import BIFWriter
from pgmpy.estimators import BayesianEstimator
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Usara el mejor modelo aprendido para inferencia


def parameter_learning(model, df):
    print("\nEstimando parámetros con BayesianEstimator...")
    estimator = BayesianEstimator(model, df)
    model.fit(df, BayesianEstimator, prior_type='BDeu')
    
    # # Guardar el modelo en la carpeta uploads con el formato BIF
    # model_file_path = os.path.join('./models', 'best_model_rb_with_parameter.bif')
    # writer = BIFWriter(model)
    # writer.write_bif(model_file_path)
    # print(f"El modelo ha sido guardado en: {model_file_path}")
    
    return model

# --- Inferencia exacta ---
def bayesian_inference(model, evidences_df, variable_name):
    print("\\nRealizando inferencia exacta para múltiples casos...")
    belief_propagation = BeliefPropagation(model)
    print("Calibrando Belief Propagation...")
    try:
        belief_propagation.calibrate()
    except Exception as e:
        print(f"[ERROR] Falló la calibración de Belief Propagation: {e}")
        # Guardar un JSON vacío o con error y retornar
        error_result = {"error": "Fallo en la calibración de Belief Propagation", "details": str(e)}
        result_file_path = os.path.join('./results', 'inferencia_rb_error_ypred_batch.json')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump([error_result], f, indent=4, ensure_ascii=False)
        print(f"Error de calibración guardado en: {result_file_path}")
        return [error_result]

    all_results = []
    num_predictions = min(1000, evidences_df.shape[0])
    print(f"Realizando inferencia para {num_predictions} casos...")

    for i in range(num_predictions):
        evidence_dict = evidences_df.iloc[i].to_dict()
        # Asegurar que los valores de evidencia sean del tipo correcto si es necesario
        for k, v in evidence_dict.items():
            if isinstance(v, float) and v.is_integer():
                evidence_dict[k] = int(v)

        print(f"  Caso {i+1}/{num_predictions} - Evidencia: {evidence_dict}") # Descomentar para debugging detallado
        try:
            result = belief_propagation.map_query(variables=[variable_name], evidence=evidence_dict)
            # Convertir tipos de numpy a tipos estándar de Python para serialización JSON
            processed_result = {k_res: int(v_res) if hasattr(v_res, 'item') else v_res for k_res, v_res in result.items()}
            all_results.append(processed_result)
        except Exception as e:
            print(f"  [ERROR] Error al procesar caso {i+1} con evidencia {evidence_dict}: {e}")
            all_results.append({"error": str(e), "evidence_case_number": i+1, "evidence_provided": evidence_dict})

    result_file_path = os.path.join('./results', 'inferencia_rb_ypred_batch.json')
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Resultados de la inferencia ({len(all_results)} casos procesados) guardados en: {result_file_path}")
    except Exception as e:
        print(f"[ERROR] No se pudieron guardar los resultados de la inferencia: {e}")
        
    return all_results


def main():
    
    model_path = './models/mejor_modelo_hill_climb_bdeu_20000_bDeuScore-5855502.80_edges_72_20250606_084240.pkl'           #Cambiar esta ruta
    print(f"Cargando el modelo desde: {model_path}")

    # 1. Leemos los DataFrames de entrenamiento y validación
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames cargados correctamente.") 

    # 2. Cargamos el mejor modelo aprendido en structure_learner.py

    

    try:
        with open(model_path, 'rb') as f:
            model_rb = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Archivo de modelo no encontrado en {model_path}")
        print("Por favor, verifica la ruta y el nombre del archivo del modelo.")
        return
    except Exception as e:
        print(f"[ERROR] Ocurrió un error al cargar el modelo: {e}")
        return
    
    # 3. Aprendizaje de parámetros usando el training set
    model_rb = parameter_learning(model_rb, train_encoded)

    #4. Evaluando modelo con Red Bayesiana
    print("\\nEvaluando el modelo con Red Bayesiana...")
    
    target_variable = 'NIVEL_DE_RIESGO_VICTIMA' # Asegúrate que esta es tu variable objetivo
    
    try:
        markov_blanket = model_rb.get_markov_blanket(target_variable)
        if not markov_blanket: # Si el target no tiene nodos en su blanket (es aislado o raíz sin hijos en el scope)
             print(f"[ADVERTENCIA] El manto de Markov para '{target_variable}' está vacío. Puede que no esté conectado o sea una variable aislada.")
             # Decide cómo proceder: quizás usar todas las variables o un subconjunto predefinido.
             # Por ahora, si está vacío, la inferencia con él no tendrá sentido.
             # Podrías usar model_rb.nodes() pero eso sería todas las variables.
             # O definir un conjunto por defecto de variables de evidencia.
             # Para este ejemplo, si está vacío, la inferencia fallará o será trivial.
             # Vamos a verificar que las columnas del blanket existan en val_encoded
        print(f"Manto de Markov de '{target_variable}': {markov_blanket}")
    except Exception as e:
        print(f"[ERROR] No se pudo obtener el Manto de Markov para '{target_variable}': {e}")
        print("Asegúrate de que la variable objetivo exista en el modelo.")
        return

    # 5. Inferencia exacta
    print("\\nPreparando datos para inferencia en lote...")
    
    # Verificar que todas las columnas del manto de Markov estén en val_encoded
    missing_cols = [col for col in markov_blanket if col not in val_encoded.columns]
    if missing_cols:
        print(f"[ERROR] Las siguientes columnas del Manto de Markov no se encuentran en val_encoded: {missing_cols}")
        print("No se puede proceder con la inferencia.")
        return

    if not markov_blanket: # Si el blanket está vacío y no se manejó antes
        print(f"[ADVERTENCIA] El Manto de Markov para '{target_variable}' está vacío. No se pueden seleccionar columnas para la evidencia.")
        # Aquí podrías decidir no hacer la inferencia o usar un conjunto de evidencia por defecto.
        # Por ahora, si está vacío, no se llamará a bayesian_inference.
        evidences_to_predict = pd.DataFrame() # DataFrame vacío
    else:
        evidences_to_predict = val_encoded[markov_blanket]
    
    if evidences_to_predict.empty and markov_blanket:
        print("[ADVERTENCIA] No hay datos en val_encoded para las columnas del Manto de Markov, o val_encoded está vacío.")
    elif not evidences_to_predict.empty:
        print(f"Se usarán {evidences_to_predict.shape[0]} filas de val_encoded para la inferencia.")
        all_results = bayesian_inference(model_rb, evidences_to_predict, target_variable)
    else:
        print("No se realizará la inferencia ya que no hay datos de evidencia preparados.")

    # Guardar metricas
    print("\\nGuardando métricas del modelo...")
    output_dir = './results'
    
    # Predicciones
    #y_train_pred = rf_model.predict(X_train)
    # Convertir all_results en una lista de valores planos
    y_val_pred = [res[target_variable] if isinstance(res, dict) and target_variable in res else None for res in all_results]

    # Filtrar casos inválidos si los hay (por errores de inferencia)
    valid_idx = [i for i, val in enumerate(y_val_pred) if val is not None]
    y_val_pred = [y_val_pred[i] for i in valid_idx]
    y_val = val_encoded[target_variable].iloc[valid_idx] 

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
    for metrics in [val_metrics]:
        print(f"\nMétricas ({metrics['dataset']}):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")

    # Guardar métricas en CSV
    metrics_df = pd.DataFrame([val_metrics])
    metrics_file_path = os.path.join(output_dir, 'metrics_rb_classic.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMétricas guardadas en: {metrics_file_path}")

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Validación')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    conf_matrix_file_path = os.path.join('./plots', 'confusion_matrix_rb_classic.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {conf_matrix_file_path}")

    # Guardar reporte de clasificación como texto
    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, 'classification_report_rb_classic.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Reporte de clasificación guardado en: {report_path}")


if __name__ == "__main__":
    main()
