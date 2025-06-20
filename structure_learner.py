from pgmpy.estimators import BayesianEstimator
import pickle
import os
import json
import time
from sklearn.model_selection import train_test_split # Added import
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MmhcEstimator, PC, GES
from pgmpy.metrics import structure_score
from pgmpy.estimators import K2, BDeu, BIC, AIC
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from pgmpy.estimators import ExpertKnowledge
import pandas as pd
import numpy as np
from datetime import datetime

# Aprender mejor estructura de un modelo bayesiano a partir de un DataFrame
# Comparar modelos con 3 tamaños de muestra diferentes.
# Usará solo el train...


def collect_all_categories(df):
    """
    Devuelve un DataFrame con al menos una fila por cada categoría en cada columna categórica.
    """
    rows = []

    for col in df.select_dtypes(include=['category']).columns:
        for cat in df[col].cat.categories:
            row = df[df[col] == cat].sample(n=1, random_state=42)
            rows.append(row)

    return pd.concat(rows).drop_duplicates().reset_index()
   
def learn_structure(df, algorithm='hill_climb', scoring_method=None, output_path=None, expert_knowledge=None, enforce_expert_knowledge=False):
        df.info()
        print("Forma de muestra del DataFrame:", df.shape)
        if algorithm == 'hill_climb':
            print(f"\\nAprendiendo con Hill Climbing usando {scoring_method}...")
            est = HillClimbSearch(df)
            if scoring_method == 'bic':
                model = est.estimate(scoring_method=BIC(df), max_iter=5000, max_indegree=5)
            elif scoring_method == 'bdeu':
                model = est.estimate(scoring_method=BDeu(df), max_indegree=5, max_iter=int(1e4))
            elif scoring_method == 'k2':
                model = est.estimate(scoring_method=K2(df), max_indegree=5, max_iter=int(1e4))
            elif scoring_method == 'bic-d':
                model = est.estimate(scoring_method=scoring_method, max_indegree=5, max_iter=int(1e4))
            else:
                raise ValueError("Scoring method no soportado para Hill Climbing.")
            bn_model = DiscreteBayesianNetwork()
            bn_model.add_nodes_from(df.columns)
            bn_model.add_edges_from(model.edges())
        elif algorithm == 'GES': #Causal Discovery
            print(f"\\nAprendiendo con GES...")
            est = GES(df)
            model = est.estimate(scoring_method=scoring_method)
            bn_model = DiscreteBayesianNetwork()
            bn_model.add_nodes_from(df.columns)
            bn_model.add_edges_from(model.edges())
        elif algorithm == 'pc':
            print(f"\\nAprendiendo con PC...")
            est = PC(df) # Initialize PC estimator once

            if scoring_method == 'pillai':
                assert not df.isnull().values.any(), "DataFrame contains NaN values"
                assert np.isfinite(df.to_numpy()).all(), "DataFrame contains inf values"
                # Validación: si hay columnas con baja varianza, eliminarlas
                low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if low_variance_cols:
                    print(f"Warning: Low-variance columns found for PC with Pillai: {low_variance_cols}")
                # --- Expert knowledge for PC ---
                model = est.estimate(
                    ci_test='pillai',
                    return_type='dag', # MODIFIED
                    max_cond_vars=5, 
                    n_jobs=-1,
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False
                )
            elif scoring_method == 'chi_square':
                model = est.estimate(
                    variant='parallel',
                    ci_test='chi_square',
                    return_type='dag', # MODIFIED from 'pdag'
                    significance_level=0.01,
                    max_cond_vars=5,  #3
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False,
                    n_jobs=-1,
                    show_progress=True
                )
            else:
                raise ValueError(f"Unsupported scoring_method '{scoring_method}' for PC algorithm.")
            
            if model is None:
                print(f"PC algorithm with {scoring_method} resulted in no edges (all variables independent). Creating an empty network with all nodes.")
                bn_model = DiscreteBayesianNetwork() # Create an empty network
                bn_model.add_nodes_from(df.columns) # Add all columns as nodes
            else:
                if not hasattr(model, 'edges'): # Defensive check
                     raise TypeError(f"Model returned by PC ({scoring_method}) is not a DAG object or similar (type: {type(model)}).")
                print("Edges encontrados:", model.edges())    
                bn_model = DiscreteBayesianNetwork()
                bn_model.add_nodes_from(model.nodes())
                bn_model.add_edges_from(model.edges())
        elif algorithm == 'mmhc':
            print("\\nAprendiendo con MMHC (Max-Min Hill Climbing)...")
            mmhc = MmhcEstimator(df)
            print("\nAprendiendo con MMHC (1.skeleton)...")
            skeleton = mmhc.mmpc()
            print("\nAprendiendo con MMHC (2.hc)...")
            hc = HillClimbSearch(df)
            model = hc.estimate(
                tabu_length=5,
                white_list=skeleton.to_directed().edges(),
                scoring_method=BDeu(df),
                max_indegree=3,
                max_iter=100
            )
            bn_model = DiscreteBayesianNetwork()
            bn_model.add_nodes_from(df.columns)
            bn_model.add_edges_from(model.edges())
        else:
            raise ValueError("Algoritmo no soportado.")
        print("Estructura aprendida:", bn_model.edges())
        return bn_model

def main():
     
    errors = []
            
    # 1. Leemos los DataFrames de entrenamiento y validación
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames cargados correctamente.")
    print("Forma de train_df:", train_df.shape)
    print("Forma de train_encoded:", train_encoded.shape)
    print("Forma de val_df:", val_df.shape)
    print("Forma de val_encoded:", val_encoded.shape)
    
    # --- Cargar dtype_definitions y re-aplicar a train_df ---
    dtype_definitions_path = './uploads/dtype_definitions.json'
    if os.path.exists(dtype_definitions_path):
        with open(dtype_definitions_path, 'r', encoding='utf-8') as f:
            dtype_definitions = json.load(f)
    #
    
    # --- Re-aplicar dtypes categóricos a sample_data para asegurar el orden ---
        for col_name, defs in dtype_definitions.items():
            if col_name in train_df.columns:
                try:
                    cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                    train_df[col_name] = train_df[col_name].astype(cat_dtype)
                except Exception as e:
                    print(f"[ADVERTENCIA] No se pudo convertir la columna {col_name} al CategoricalDtype especificado: {e}")
                    print(f"  Categorías esperadas: {defs['categories']}")
                    print(f"  Categorías encontradas en los datos: {list(train_df[col_name].unique()) if hasattr(train_df[col_name], 'unique') else 'N/A'}")
        # --- Fin de la re-aplicación ---
    
    algorithms_to_experiment = [
        ('hill_climb', 'bic-d'), 
        ('hill_climb', 'k2'),
        ('hill_climb', 'bdeu'),
        ('pc', 'pillai'),
        ('pc', 'chi_square'),
	    ('GES','bic-d'),
        ('GES', 'bic-cg')
    ]
    
    size_df = train_df.shape[0]
    
    sample_sizes = [10000, 20000, 50000, 100000, 150000, 200000, size_df]
    results = []
    trained_models = {}

    expert_knowledge = {
        'forbidden_edges': [],
        'required_edges': [('ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA')]
    }
    
    for sample_size in sample_sizes:
        # Paso 1: obtener una muestra mínima que cubra todas las categorías
        df_sample_min = collect_all_categories(train_df)
        min_indices_sample = set(df_sample_min['index'])  # índices ya seleccionados
        df_sample_min = df_sample_min.set_index('index')

        if sample_size < len(df_sample_min):
            print(f"[WARNING] Sample size {sample_size} is menor que el mínimo necesario {len(df_sample_min)}. Se omite esta muestra.")
            continue

        # Paso 2: completar aleatoriamente desde train_df hasta el tamaño deseado
        n_extra = sample_size - len(df_sample_min)
        df_remaining_sample = train_df.drop(index=min_indices_sample)
        df_sample_extra = df_remaining_sample.sample(n=n_extra, random_state=42)

        # Formar conjunto final de muestra
        sample_df = pd.concat([df_sample_min, df_sample_extra])
        sample_indices = sample_df.index

        sample_data = train_df.loc[sample_indices].reset_index(drop=True)
        sample_data_encoded = train_encoded.loc[sample_indices].reset_index(drop=True)

        # --- Re-aplicar dtypes categóricos a sample_data para asegurar el orden ---
        for col_name, defs in dtype_definitions.items():
            if col_name in sample_data.columns:
                try:
                    cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                    sample_data[col_name] = sample_data[col_name].astype(cat_dtype)
                except Exception as e:
                    print(f"[ADVERTENCIA] No se pudo convertir la columna {col_name} al CategoricalDtype especificado: {e}")
                    print(f"  Categorías esperadas: {defs['categories']}")
                    print(f"  Categorías encontradas en los datos: {list(sample_data[col_name].unique()) if hasattr(sample_data[col_name], 'unique') else 'N/A'}")
        # --- Fin de la re-aplicación ---

        for algorithm, score_method in algorithms_to_experiment:
            if algorithm == 'hill_climb' or algorithm == 'pc':
                df_to_sl=sample_data_encoded
            else:
                df_to_sl=sample_data
                
            # Validación: si es PC y sample_size == 50000, saltar este experimento    
            if algorithm == 'pc' and sample_size > 100000: 
                print (f"[AVISO] se omite PC con sample_size>100000") # Adjusted message to reflect 50000
                continue
            
            expert_knowledge = None
            enforce_expert_knowledge  = False
            if algorithm == 'pc':
                expert_knowledge  = ExpertKnowledge(
                    required_edges=[
                        ('ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA')
                    ],
                    forbidden_edges=[
                    ]
                )
                enforce_expert_knowledge = True
            
            # # Debugging for PC Pillai specifically for the failing case
            # if algorithm == 'pc' and score_method == 'pillai' and sample_size == 10000:
            #     print("\\n--- DEBUG INFO FOR PC PILLAI (sample_size=10000) ---")
            #     problematic_vars = ['ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA']
            #     for var_name in problematic_vars:
            #         if var_name in df_to_sl.columns:
            #             print(f"Value counts for {var_name} in df_to_sl (shape: {df_to_sl.shape}):")
            #             print(df_to_sl[var_name].value_counts(dropna=False))
            #             print(f"Description for {var_name}:")
            #             print(df_to_sl[var_name].describe())
            #             print(f"Is {var_name} all NaN? {df_to_sl[var_name].isnull().all()}")
            #             print(f"Number of unique values for {var_name}: {df_to_sl[var_name].nunique()}")
            #         else:
            #             print(f"Variable {var_name} not in df_to_sl.columns")
            #     # print(f"Expert knowledge for this run: {active_expert_knowledge.rules if active_expert_knowledge else 'None'}") # Old line
            #     if active_expert_knowledge:
            #         print(f"Expert knowledge for this run: Required Edges: {active_expert_knowledge.required_edges}, Forbidden Edges: {active_expert_knowledge.forbidden_edges}")
            #     else:
            #         print("Expert knowledge for this run: None")
            #     print(f"Enforce expert knowledge: {active_enforce_expert_knowledge}")
            #     print("--- END DEBUG INFO ---\\n")

            print(f"\\nAprendiendo estructura con {algorithm} ({score_method}) with sample size = {sample_size}...")
            start_time = time.time()
            
            try:
                model = learn_structure(
                    df_to_sl,
                    algorithm=algorithm,
                    scoring_method=score_method,
                    # output_path=f'./models/model_structure_31_{algorithm}_{score_method}_{sample_size}.pkl', # User has this commented
                    expert_knowledge=expert_knowledge,
                    enforce_expert_knowledge=enforce_expert_knowledge
                )
                model_variables = set(var for edge in model.edges() for var in edge)
                # Use train_df_encoded for calculating structure score
                df_filtered = train_encoded[list(model_variables)] 
                score = structure_score(model, df_filtered, scoring_method="bdeu")
                print("Calidad de red BDeue:", score)
                elapsed_time = time.time() - start_time
                key = f"{algorithm}_{score_method}_{sample_size}"
                trained_models[key] = model
                results.append({'Model': model,
                                'BDeu_Score': score,
                                'Score_method': score_method,
                                'Algorithm': algorithm,
                                'Sample_Size': sample_size,
                                'Training_Time_Seconds': elapsed_time,
                                'Number_of_Edges': len(model),
                                'Number_of_df_variables': len(df_to_sl.columns)
                                })
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Falló {algorithm} con {score_method} y sample_size={sample_size}: {error_msg}")
                errors.append({
                    'algorithm': algorithm,
                    'score_method': score_method,
                    'sample_size': sample_size,
                    'error': error_msg
                })
                continue
            
    results_structure_learning = pd.DataFrame(results)
    results_structure_learning = results_structure_learning.sort_values(by='BDeu_Score', ascending=False).reset_index(drop=True)
    
    print("\\nTabla comparativa de resultados:")
    print(results_structure_learning.to_string(index=False))
    comparison_file_path = os.path.join('./results', 'resultados_rb_classic.csv')
    results_structure_learning.to_csv(comparison_file_path, index=False)
    print(f"Resultados guardados en: {comparison_file_path}")
    
    # --- Guardar el mejor modelo ---
    if not results_structure_learning.empty:
        # Filtrar modelos según la condición para 'pc'
        filtered_results = results_structure_learning.copy()
        for idx, row in filtered_results.iterrows():
            if row['Algorithm'] == 'pc' and row['Number_of_Edges'] != row['Number_of_df_variables']:
                print(f"[INFO] Modelo PC con sample_size={row['Sample_Size']} descartado porque Number_of_Edges ({row['Number_of_Edges']}) != Number_of_df_variables ({row['Number_of_df_variables']})")
                filtered_results = filtered_results.drop(idx)
        filtered_results = filtered_results.reset_index(drop=True)

        if not filtered_results.empty:
            best_row = filtered_results.iloc[0]
            best_model_key = f"{best_row['Algorithm']}_{best_row['Score_method']}_{int(best_row['Sample_Size'])}"
            best_score = best_row['BDeu_Score']
            best_model_edges = len(trained_models[best_model_key].edges()) if best_model_key in trained_models and hasattr(trained_models[best_model_key], 'edges') else 'N/A'
            best_model = trained_models[best_model_key]
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            filename_best = f"./models/mejor_modelo_{best_model_key}_bDeuScore{best_score:.2f}_edges_{best_model_edges}_{timestamp_str}.pkl"
            with open(filename_best, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"\nEl mejor modelo ha sido guardado en: {filename_best}")

            # --- Guardar el segundo mejor modelo ---
            if len(filtered_results) > 1:
                second_best_row = filtered_results.iloc[1]
                second_best_model_key = f"{second_best_row['Algorithm']}_{second_best_row['Score_method']}_{int(second_best_row['Sample_Size'])}"
                second_best_score = second_best_row['BDeu_Score']
                second_best_model_edges = len(trained_models[second_best_model_key].edges()) if second_best_model_key in trained_models and hasattr(trained_models[second_best_model_key], 'edges') else 'N/A'
                if second_best_model_key in trained_models:
                    second_best_model = trained_models[second_best_model_key]
                    filename_second_best = f"./models/segundo_mejor_modelo_{second_best_model_key}_bDeuScore{second_best_score:.2f}_edges_{second_best_model_edges}_{timestamp_str}.pkl"
                    with open(filename_second_best, 'wb') as f:
                        pickle.dump(second_best_model, f)
                    print(f"El segundo mejor modelo ha sido guardado en: {filename_second_best}")
                else:
                    print("[ADVERTENCIA] No se encontró el segundo mejor modelo en trained_models.")
            else:
                print("No hay un segundo mejor modelo para guardar.")
        else:
            print("[ADVERTENCIA] No hay modelos válidos tras filtrar por la condición de PC. No se puede guardar el mejor modelo ni el segundo mejor.")
            best_model = None
    else:
        print("[ADVERTENCIA] No se entrenaron modelos, no se puede guardar el mejor modelo ni el segundo mejor.")
        best_model = None # Ensure best_model is None if no models were trained
    
    # Guardar errores a CSV
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv('./uploads/structure_learning_classic_errors.csv', index=False)
        print(f"[INFO] Errores guardados en './uploads/structure_learning_classic_errors.csv'")
        
    # Guardar imagen del best_model en la carpeta dag
    try:
        if best_model: # Check if best_model is not None
            # Convertir el modelo a un grafo de networkx
            # nx_graph = best_model.to_networkx() # Original line causing AttributeError
            
            # Workaround: Manually create networkx.DiGraph
            nx_graph = nx.DiGraph()
            if hasattr(best_model, 'nodes') and hasattr(best_model, 'edges'):
                nx_graph.add_nodes_from(best_model.nodes())
                nx_graph.add_edges_from(best_model.edges())
            else:
                # This case should ideally not be reached if best_model is a valid pgmpy model
                print("[ERROR] best_model object does not have nodes() or edges() methods.")
                raise TypeError("best_model cannot be converted to a networkx graph.")

            # Crear el objeto pydot
            pydot_graph = to_pydot(nx_graph)
            
            # Ensure the ./dag directory exists
            os.makedirs('./dag', exist_ok=True)
            
            # Guardar como imagen PNG
            pydot_graph.write_png('./dag/best_model_rb_classic.png')
            print('Imagen del best_model guardada en ./dag/best_model.png')
        else:
            print("[ADVERTENCIA] No se guardó imagen del modelo porque no se entrenó ningún modelo exitosamente.")
    except Exception as e:
        print(f'No se pudo guardar la imagen del modelo: {e}')
    
    
    #Markov
    if best_model: # Check if best_model is not None
        target_variable = 'NIVEL_DE_RIESGO_VICTIMA'
        if target_variable in best_model.nodes():
            markov_blanket = best_model.get_markov_blanket(target_variable)
            print(f"Markov Blanket de '{target_variable}':", markov_blanket)
        else:
            print(f"[ADVERTENCIA] La variable '{target_variable}' no se encuentra en los nodos del mejor modelo. No se puede calcular el Markov Blanket.")
    else:
        print("[ADVERTENCIA] No se puede calcular el Markov Blanket porque no se entrenó ningún modelo exitosamente.")
    
    # Inferencia exacta
    
    #Las evidencias son val_df_encoded, extrae solo las columnas que se obtienen en markov_blanket
    # evidence = val_encoded[markov_blanket].iloc[0].to_dict()
    # print("Evidencia para la inferencia:", evidence)
    

    #bayesian_inference(model_rb, evidence)

if __name__ == "__main__":
    main()
