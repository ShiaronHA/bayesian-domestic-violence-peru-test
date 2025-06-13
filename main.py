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
from pgmpy.sampling import GibbsSampling


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
def bayesian_inference_exact(model, evidences_df, variable_name):
    result_file_path = os.path.join('./results', 'inferencia_exact_rb_ypred_batch.json')
    calibration_error_file_path = os.path.join('./results', 'inferencia_rb_error_ypred_batch.json')

    print("\\\\nRealizando inferencia exacta para múltiples casos...")
    belief_propagation = BeliefPropagation(model)
    print("Calibrando Belief Propagation...")
    try:
        belief_propagation.calibrate()
    except Exception as e:
        print(f"[ERROR] Falló la calibración de Belief Propagation: {e}")
        error_result = {"error": "Fallo en la calibración de Belief Propagation", "details": str(e)}
        # Save calibration error to its specific file
        with open(calibration_error_file_path, 'w', encoding='utf-8') as f:
            json.dump([error_result], f, indent=4, ensure_ascii=False)
        print(f"Error de calibración guardado en: {calibration_error_file_path}")
        return [error_result] # Return the error, main loop will handle this

    all_results = []
    start_case_index = 0

    # Try to load previous results
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list) and all(isinstance(item, dict) for item in loaded_data):
                # Check if the loaded data is actually a calibration error saved to the wrong file
                is_calibration_error = False
                if len(loaded_data) == 1 and "error" in loaded_data[0]:
                    if "Fallo en la calibración de Belief Propagation" in loaded_data[0]["error"]:
                        is_calibration_error = True
                
                if not is_calibration_error:
                    all_results = loaded_data
                    start_case_index = len(all_results)
                    print(f"Se cargaron {start_case_index} resultados previos desde '{result_file_path}'.")
                else:
                    print(f"El archivo '{result_file_path}' contenía un error de calibración. Iniciando inferencia de resultados de nuevo.")
            else:
                print(f"[ADVERTENCIA] El archivo '{result_file_path}' no contiene una lista de resultados válida. Iniciando de nuevo.")
        except json.JSONDecodeError:
            print(f"[ADVERTENCIA] Error al decodificar JSON desde '{result_file_path}'. Iniciando de nuevo.")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudieron cargar resultados previos desde '{result_file_path}': {e}. Iniciando de nuevo.")

    total_max_predictions = 100
    batch_size = 7
    num_total_to_process = min(total_max_predictions, evidences_df.shape[0])

    if start_case_index >= num_total_to_process:
        print(f"Todos los {num_total_to_process} casos ya han sido procesados según el archivo '{result_file_path}'.")
        return all_results

    print(f"Realizando inferencia para un total de {num_total_to_process} casos, en lotes de hasta {batch_size}.")
    if start_case_index > 0:
        print(f"Reanudando desde el caso {start_case_index + 1}.")

    for batch_start_idx in range(0, num_total_to_process, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, num_total_to_process)
        
        if batch_end_idx <= start_case_index:
            print(f"  Lote de casos {batch_start_idx + 1} a {batch_end_idx} ya procesado y guardado. Saltando.")
            continue

        current_batch_df = evidences_df.iloc[batch_start_idx:batch_end_idx]
        
        if current_batch_df.empty:
            continue

        print(f"  Procesando lote: casos {batch_start_idx + 1} a {batch_end_idx} (de {num_total_to_process} en total)...")

        for i in range(current_batch_df.shape[0]):
            actual_case_index_in_original_df = batch_start_idx + i

            if actual_case_index_in_original_df < start_case_index:
                continue

            evidence_dict = current_batch_df.iloc[i].to_dict()
            
            # Ensure evidence_dict values are JSON serializable (especially for error logging)
            serializable_evidence_dict = {}
            for k, v in evidence_dict.items():
                if isinstance(v, float) and v.is_integer():
                    serializable_evidence_dict[k] = int(v)
                elif isinstance(v, (np.integer, np.int64)):
                    serializable_evidence_dict[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    serializable_evidence_dict[k] = float(v)
                elif pd.isna(v): # Represent NaN as None for JSON
                    serializable_evidence_dict[k] = None
                else:
                    serializable_evidence_dict[k] = v
            
            print(f"    Caso {actual_case_index_in_original_df + 1}/{num_total_to_process} - Evidencia: {serializable_evidence_dict}")
            try:
                # Use original evidence_dict for pgmpy, serializable_evidence_dict for logging
                result = belief_propagation.map_query(variables=[variable_name], evidence=evidence_dict)
                processed_result = {k_res: int(v_res) if hasattr(v_res, 'item') else v_res for k_res, v_res in result.items()}
                all_results.append(processed_result)
            except Exception as e:
                error_message = f"Error al procesar caso {actual_case_index_in_original_df + 1}"
                print(f"    [ERROR] {error_message} con evidencia {serializable_evidence_dict}: {e}")
                all_results.append({"error": str(e), 
                                    "details": error_message,
                                    "evidence_case_number": actual_case_index_in_original_df + 1, 
                                    "evidence_provided": serializable_evidence_dict})
            
            # Save after each case (successful or error)
            try:
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                print(f"      Progreso guardado tras caso {actual_case_index_in_original_df + 1}. Total {len(all_results)} items en '{result_file_path}'.")
            except Exception as e:
                print(f"      [ERROR AL GUARDAR] No se pudo guardar el progreso tras caso {actual_case_index_in_original_df + 1}: {e}")

    # Final confirmation message
    if start_case_index < num_total_to_process : # Only if some processing was attempted in this run
        print(f"Proceso de inferencia exacta completado. Total de resultados ({len(all_results)} casos) guardados en: {result_file_path}")
    
    return all_results

#Inferencia aproximada con Variational Inference, gibbs sampling

def bayesian_inference_approximate(model, evidences_df, variable_name, n_samples=1000, seed=None):
    print("\\\\nRealizando inferencia aproximada con Gibbs Sampling para múltiples casos...")
    gibbs_sampler = GibbsSampling(model)
    
    all_results = []
    total_max_predictions = 1000  # Límite superior de predicciones a intentar
    batch_size = 7
    
    num_total_to_process = min(total_max_predictions, evidences_df.shape[0])
        
    print(f"Realizando inferencia aproximada para un total de {num_total_to_process} casos, en lotes de hasta {batch_size} (muestras por caso: {n_samples})...")

    for batch_start_idx in range(0, num_total_to_process, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, num_total_to_process)
        current_batch_df = evidences_df.iloc[batch_start_idx:batch_end_idx]

        if current_batch_df.empty:
            continue
            
        print(f"  Procesando lote (aproximado): casos {batch_start_idx + 1} a {batch_end_idx} (de {num_total_to_process} en total)...")

        for i in range(current_batch_df.shape[0]):
            actual_case_index_in_original_df = batch_start_idx + i
            evidence_dict = current_batch_df.iloc[i].to_dict()
            # Asegurar que los valores de evidencia sean del tipo correcto
            keys_to_delete = [] # Para evitar modificar el diccionario mientras se itera
            for k, v in evidence_dict.items():
                if pd.isna(v): # Remove NaN values from evidence
                    keys_to_delete.append(k)
                    continue
                if isinstance(v, float) and v.is_integer():
                    evidence_dict[k] = int(v)
            for k_del in keys_to_delete:
                del evidence_dict[k_del]
            
            print(f"    Caso {actual_case_index_in_original_df + 1}/{num_total_to_process} - Evidencia (aprox): {evidence_dict}")
            try:
                # Perform sampling using GibbsSampling
                # Set show_progress=False for less verbose output during batch processing
                samples_df = gibbs_sampler.sample(evidence=evidence_dict, size=n_samples, seed=seed, show_progress=False) 
                
                if variable_name in samples_df.columns:
                    # Calculate mode of the target variable column from the samples
                    predicted_value_series = samples_df[variable_name].mode()
                    
                    if not predicted_value_series.empty:
                        # mode() can return multiple values if they have the same frequency. Take the first.
                        result_value = predicted_value_series[0] 
                        
                        # Ensure it's a standard Python type for JSON serialization
                        if hasattr(result_value, 'item'): # Handles numpy types like numpy.int64
                            result_value = result_value.item()
                        all_results.append({variable_name: result_value})
                    else:
                        print(f"    [ADVERTENCIA] No se pudo determinar el modo para '{variable_name}' en el caso {actual_case_index_in_original_df + 1}. La serie de modos está vacía.")
                        all_results.append({
                            "error": f"No mode found for {variable_name} (empty mode series)", 
                            "evidence_case_number": actual_case_index_in_original_df + 1, 
                            "evidence_provided": evidence_dict
                        })
                else:
                    print(f"    [ERROR] La variable '{variable_name}' no se encontró en las muestras generadas para el caso {actual_case_index_in_original_df + 1}.")
                    all_results.append({
                        "error": f"Variable {variable_name} not in generated samples", 
                        "evidence_case_number": actual_case_index_in_original_df + 1, 
                        "evidence_provided": evidence_dict
                    })

            except Exception as e:
                print(f"    [ERROR] Error al procesar caso {actual_case_index_in_original_df + 1} con evidencia {evidence_dict} usando Gibbs Sampling: {e}")
                all_results.append({"error": str(e), "evidence_case_number": actual_case_index_in_original_df + 1, "evidence_provided": evidence_dict})

    result_file_path = os.path.join('./results', 'inferencia_approx_gibbs_ypred_batch.json')
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Resultados de la inferencia aproximada ({len(all_results)} casos procesados) guardados en: {result_file_path}")
    except Exception as e:
        print(f"[ERROR] No se pudieron guardar los resultados de la inferencia aproximada: {e}")
    
    return all_results

def metrics_to_dataframe(y_val, y_val_pred, type_inference, model, output_dir='./results'):
    
    # Métricas para val
    val_metrics = {
        'model': model,
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
    metrics_file_path = os.path.join(output_dir, f'metrics_rb_classic_{model}_{type_inference}.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMétricas guardadas en: {metrics_file_path}")

    # Matriz de confusión
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Validación')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    conf_matrix_file_path = os.path.join('./plots', f'confusion_matrix_rb_classic_{model}_{type_inference}.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {conf_matrix_file_path}")

    # Guardar reporte de clasificación como texto
    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, f'classification_report_rb_classic_{model}_{type_inference}.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Reporte de clasificación guardado en: {report_path}")
    
def main():
    
    #model_path = './models/mejor_modelo_hill_climb_bdeu_20000_bDeuScore-5855502.80_edges_72_20250606_084240.pkl'   
    #model_path = './models/mejor_modelo_hill_climb_bic-d_330504_bDeuScore-5747335.12_edges_103_20250608_214125.pkl'
    model_path = './models/dag_aprendido_with_llm_gemini.pkl'
    model = 'gemini'        
    print(f"Cargando el modelo desde: {model_path}")

    # 1. Leemos los DataFrames de entrenamiento y validación
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames cargados correctamente.") 
    
    # 2. Borrar nas
    print("\n[INFO] Conteo de NaNs en cada columna del DataFrame de entrenamiento:")
    print(train_encoded.isna().sum())
    train_encoded.dropna(inplace=True)
    val_encoded.dropna(inplace=True)

    # 3. Cargamos el mejor modelo aprendido en structure_learner.py

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

    # Lista de nodos a eliminar si el modelo es 'gemini'
    nodos_a_excluir = ["TRATAMIENTO_VICTIMA", "VIOLENCIA_ECONOMICA"]

    if model == "gemini":
        for nodo in nodos_a_excluir:
            if nodo in model_rb.nodes():
                print(f"Excluyendo nodo {nodo} del modelo gemini...")
                try:
                    model_rb.remove_cpds(model_rb.get_cpds(nodo))
                except:
                    pass  # Si no tiene CPD
                model_rb.remove_node(nodo)
                
        # Ajustar DataFrames y evidencias
        val_encoded = val_encoded.drop(columns=nodos_a_excluir, errors='ignore')
        #evidences_to_predict = [
        #    {k: v for k, v in ev.items() if k not in nodos_a_excluir}
        #    for ev in evidences_to_predict
        #]
    
    #4. Evaluando modelo con Red Bayesiana
    print("\\nEvaluando el modelo con Red Bayesiana...")
    
    target_variable = 'NIVEL_DE_RIESGO_VICTIMA' #Variable objetivo
    
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

    


    # Verificar que todas las columnas del manto de Markov estén en val_encoded
    missing_cols = [col for col in markov_blanket if col not in val_encoded.columns]
    if missing_cols:
        print(f"[ERROR] Las siguientes columnas del Manto de Markov no se encuentran en val_encoded: {missing_cols}")
        print("No se puede proceder con la inferencia.")
        return

    if not markov_blanket: # Si el blanket está vacío y no se manejó antes
        print(f"[ADVERTENCIA] El Manto de Markov para '{target_variable}' está vacío. No se pueden seleccionar columnas para la evidencia.")
        evidences_to_predict = pd.DataFrame() # DataFrame vacío
    else:
        evidences_to_predict = val_encoded[markov_blanket]

    if model == "gemini":
        evidences_to_predict = [
            {k: v for k, v in ev.to_dict().items() if k not in nodos_a_excluir}
            for _, ev in evidences_to_predict.iterrows()
        ] 
        
    # 3. Aprendizaje de parámetros usando el training set
    model_rb = parameter_learning(model_rb, train_encoded)


    output_dir = './results'

    
    # 5. Inferencia exacta
    print("\\nPreparando datos para inferencia en lote...")

    type_inference = ['Exact']#, 'Approximate']
    
    for i in type_inference:
        print(f"\nGuardando métricas para tipo de inferencia: {i}")
        if i == 'Exact':
            if evidences_to_predict.empty and markov_blanket:
                print("[ADVERTENCIA] No hay datos en val_encoded para las columnas del Manto de Markov, o val_encoded está vacío.")
            elif not evidences_to_predict.empty:
                print(f"Se usarán {evidences_to_predict.shape[0]} filas de val_encoded para la inferencia.")
                all_results = bayesian_inference_exact(model_rb, evidences_to_predict, target_variable)

                # Limitar a los primeros 100 resultados
                all_results = all_results[:100]
            else:
                print("No se realizará la inferencia ya que no hay datos de evidencia preparados.")
                continue  # Saltar a la siguiente iteración si no hay resultados

            # Predicciones
            y_val_pred = [res[target_variable] if isinstance(res, dict) and target_variable in res else None for res in all_results]

            # Filtrar casos inválidos si los hay (por errores de inferencia)
            valid_idx = [i for i, val in enumerate(y_val_pred) if val is not None]
            y_val_pred = [y_val_pred[i] for i in valid_idx]

            # Obtener los valores reales correspondientes y limitar a 100 registros
            y_val = val_encoded[target_variable].iloc[valid_idx]
            y_val = y_val.iloc[:100]

            # Guardar métricas en DataFrame
            print("\nGuardando métricas del modelo...")
            metrics_to_dataframe(y_val, y_val_pred, i, model, output_dir)

        elif i == 'Approximate':
            if evidences_to_predict.empty and markov_blanket:
                print("[ADVERTENCIA] No hay datos en val_encoded para las columnas del Manto de Markov, o val_encoded está vacío.")
            elif not evidences_to_predict.empty:
                print(f"Se usarán {evidences_to_predict.shape[0]} filas de val_encoded para la inferencia.")
                all_results = bayesian_inference_approximate(model_rb, evidences_to_predict, target_variable)
            else:
                print("No se realizará la inferencia ya que no hay datos de evidencia preparados.")
                
            # Predicciones
            # Convertir all_results en una lista de valores planos
            y_val_pred = [res[target_variable] if isinstance(res, dict) and target_variable in res else None for res in all_results]

            # Filtrar casos inválidos si los hay (por errores de inferencia)
            valid_idx = [i for i, val in enumerate(y_val_pred) if val is not None]
            y_val_pred = [y_val_pred[i] for i in valid_idx]
            y_val = val_encoded[target_variable].iloc[valid_idx]
            
            print("\\nGuardando métricas del modelo...")
            metrics_to_dataframe(y_val, y_val_pred, i, model, output_dir)

if __name__ == "__main__":
    main()
