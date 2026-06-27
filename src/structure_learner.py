import pickle
import os
import json
import time
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MmhcEstimator, PC, GES
from pgmpy.metrics import structure_score
from pgmpy.estimators import K2, BDeu, BIC
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from pgmpy.estimators import ExpertKnowledge
import pandas as pd
import numpy as np
from datetime import datetime

# --- Paths ---
TRAIN_ENCODED_PATH = './datasets/train_encoded.csv'
TRAIN_DF_PATH      = './datasets/train_df.csv'
VAL_ENCODED_PATH   = './datasets/val_encoded.csv'
VAL_DF_PATH        = './datasets/val_df.csv'
DTYPE_DEFS_PATH    = './uploads/dtype_definitions.json'
RESULTS_PATH       = './results/resultados_rb_classic.csv'
ERRORS_PATH        = './uploads/structure_learning_classic_errors.csv'
DAG_IMAGE_PATH     = './dag/best_model_rb_classic.png'

# --- Experiment configuration ---
SAMPLE_SIZES = [10000, 20000, 50000, 100000, 150000, 200000] 
ALGORITHMS = [
    ('hill_climb', 'bic-d'),
    ('hill_climb', 'bdeu'),
    ('pc',         'pillai'),
    ('pc',         'chi_square'),
    ('GES',        'bic-d'),
    ('GES',        'bic-cg'),
]
MAX_INDEGREE    = 5
MAX_ITER        = int(1e4)
PC_MAX_COND_VARS = 5
PC_SIGNIFICANCE  = 0.01
RANDOM_STATE     = 42

# Required edge based on domain knowledge: ethnicity determines mother tongue
REQUIRED_EDGES = [('ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA')]
TARGET_VARIABLE = 'NIVEL_DE_RIESGO_VICTIMA'


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
   
def learn_structure(df, algorithm='hill_climb', scoring_method=None, expert_knowledge=None, enforce_expert_knowledge=False):
        print("Sample DataFrame shape:", df.shape)
        if algorithm == 'hill_climb':
            print(f"\nLearning with Hill Climbing using {scoring_method}...")
            est = HillClimbSearch(df)
            if scoring_method == 'bic':
                model = est.estimate(scoring_method=BIC(df), max_iter=MAX_ITER, max_indegree=MAX_INDEGREE)
            elif scoring_method == 'bdeu':
                model = est.estimate(scoring_method=BDeu(df), max_indegree=MAX_INDEGREE, max_iter=MAX_ITER)
            elif scoring_method == 'k2':
                model = est.estimate(scoring_method=K2(df), max_indegree=MAX_INDEGREE, max_iter=MAX_ITER)
            elif scoring_method == 'bic-d':
                model = est.estimate(scoring_method=scoring_method, max_indegree=MAX_INDEGREE, max_iter=MAX_ITER)
            else:
                raise ValueError("Scoring method no soportado para Hill Climbing.")
            bn_model = DiscreteBayesianNetwork()
            bn_model.add_nodes_from(df.columns)
            bn_model.add_edges_from(model.edges())
        elif algorithm == 'GES': #Causal Discovery
            print(f"\nLearning with GES...")
            est = GES(df)
            model = est.estimate(scoring_method=scoring_method)
            bn_model = DiscreteBayesianNetwork()
            bn_model.add_nodes_from(df.columns)
            bn_model.add_edges_from(model.edges())
        elif algorithm == 'pc':
            print(f"\nLearning with PC...")
            est = PC(df) # Initialize PC estimator once

            if scoring_method == 'pillai':
                assert not df.isnull().values.any(), "DataFrame contains NaN values"
                assert np.isfinite(df.to_numpy()).all(), "DataFrame contains inf values"
                # Validate: drop low-variance columns
                low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if low_variance_cols:
                    print(f"Warning: Low-variance columns found for PC with Pillai: {low_variance_cols}")
                # --- Expert knowledge for PC ---
                model = est.estimate(
                    ci_test='pillai',
                    return_type='dag',
                    max_cond_vars=PC_MAX_COND_VARS,
                    n_jobs=-1,
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False
                )
            elif scoring_method == 'chi_square':
                model = est.estimate(
                    variant='parallel',
                    ci_test='chi_square',
                    return_type='dag',
                    significance_level=PC_SIGNIFICANCE,
                    max_cond_vars=PC_MAX_COND_VARS,
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False,
                    n_jobs=-1,
                    show_progress=True
                )
            else:
                raise ValueError(f"Unsupported scoring_method '{scoring_method}' for PC algorithm.")
            
            if model is None:
                print(f"PC algorithm with {scoring_method} resulted in no edges (all variables independent). Creating an empty network with all nodes.")
                bn_model = DiscreteBayesianNetwork() 
                bn_model.add_nodes_from(df.columns) 
            else:
                if not hasattr(model, 'edges'): # Defensive check
                     raise TypeError(f"Model returned by PC ({scoring_method}) is not a DAG object or similar (type: {type(model)}).")
                print("Edges found:", model.edges())    
                bn_model = DiscreteBayesianNetwork()
                bn_model.add_nodes_from(model.nodes())
                bn_model.add_edges_from(model.edges())
        elif algorithm == 'mmhc':
            print("\nLearning with MMHC (Max-Min Hill Climbing)...")
            mmhc = MmhcEstimator(df)
            print("\nLearning MMHC step 1: skeleton...")
            skeleton = mmhc.mmpc()
            print("\nLearning MMHC step 2: hill climb...")
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
        print("Learned structure:", bn_model.edges())
        return bn_model

def main():
     
    errors = []
            
    train_encoded = pd.read_csv(TRAIN_ENCODED_PATH)
    train_df      = pd.read_csv(TRAIN_DF_PATH)
    val_encoded   = pd.read_csv(VAL_ENCODED_PATH)
    val_df        = pd.read_csv(VAL_DF_PATH)
    print("DataFrames loaded successfully.")
    print("train_df shape:", train_df.shape)
    print("train_encoded shape:", train_encoded.shape)
    print("val_df shape:", val_df.shape)
    print("val_encoded shape:", val_encoded.shape)
    
    # Re-apply categorical dtypes to train_df to ensure category order
    dtype_definitions_path = DTYPE_DEFS_PATH
    if os.path.exists(dtype_definitions_path):
        with open(dtype_definitions_path, 'r', encoding='utf-8') as f:
            dtype_definitions = json.load(f)
        for col_name, defs in dtype_definitions.items():
            if col_name in train_df.columns:
                try:
                    cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                    train_df[col_name] = train_df[col_name].astype(cat_dtype)
                except Exception as e:
                    print(f"[WARNING] Could not convert column '{col_name}' to specified CategoricalDtype: {e}")
                    print(f"  Expected categories: {defs['categories']}")
                    print(f"  Categories found in data: {list(train_df[col_name].unique()) if hasattr(train_df[col_name], 'unique') else 'N/A'}")
    
    sample_sizes = SAMPLE_SIZES + [train_df.shape[0]]
    results = []
    trained_models = {}

    for sample_size in sample_sizes:
        # Step 1: get minimum sample covering all categories
        df_sample_min = collect_all_categories(train_df)  # guarantees all categories are represented
        min_indices_sample = set(df_sample_min['index'])
        df_sample_min = df_sample_min.set_index('index')

        if sample_size < len(df_sample_min):
            print(f"[WARNING] Sample size {sample_size} is smaller than the minimum required ({len(df_sample_min)}). Skipping.")
            continue

        # Step 2: fill remaining rows randomly up to desired sample size
        n_extra = sample_size - len(df_sample_min)
        df_remaining_sample = train_df.drop(index=min_indices_sample)
        df_sample_extra = df_remaining_sample.sample(n=n_extra, random_state=RANDOM_STATE)

        sample_df = pd.concat([df_sample_min, df_sample_extra])
        sample_indices = sample_df.index

        sample_data = train_df.loc[sample_indices].reset_index(drop=True)
        sample_data_encoded = train_encoded.loc[sample_indices].reset_index(drop=True)

        # --- Re-apply categorical dtypes to sample_data to ensure category order ---
        for col_name, defs in dtype_definitions.items():
            if col_name in sample_data.columns:
                try:
                    cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                    sample_data[col_name] = sample_data[col_name].astype(cat_dtype)
                except Exception as e:
                    print(f"[WARNING] Could not convert column '{col_name}' to specified CategoricalDtype: {e}")
                    print(f"  Expected categories: {defs['categories']}")
                    print(f"  Categories found in data: {list(sample_data[col_name].unique()) if hasattr(sample_data[col_name], 'unique') else 'N/A'}")
        # --- End re-apply ---

        for algorithm, score_method in ALGORITHMS:
            if algorithm == 'hill_climb' or algorithm == 'pc':
                df_to_sl=sample_data_encoded
            else:
                df_to_sl=sample_data
                
        # Note: PC with large sample sizes can be very slow
            # if algorithm == 'pc' and sample_size > 100000: 
            #     print (f"[AVISO] se omite PC con sample_size>100000") # Adjusted message to reflect 50000
            #     continue
            
            expert_knowledge = None
            enforce_expert_knowledge  = False
            if algorithm == 'pc':
                expert_knowledge = ExpertKnowledge(
                    required_edges=REQUIRED_EDGES,
                    forbidden_edges=[]
                )
                enforce_expert_knowledge = True
            
            print(f"\nLearning structure with {algorithm} ({score_method}), sample size = {sample_size}...")
            start_time = time.time()
            
            try:
                model = learn_structure(
                    df_to_sl,
                    algorithm=algorithm,
                    scoring_method=score_method,
                    expert_knowledge=expert_knowledge,
                    enforce_expert_knowledge=enforce_expert_knowledge
                )
                model_variables = set(var for edge in model.edges() for var in edge)
                # Ensure all model nodes are present (important for PC)
                for col in df_to_sl.columns:
                    if col not in model.nodes():
                        model.add_node(col)
                # Para el score, si faltan columnas en model_variables, usa todas las columnas del DataFrame
                if len(model_variables) == 0:
                    df_filtered = train_encoded[df_to_sl.columns]
                else:
                    missing_vars = set(df_to_sl.columns) - model_variables
                    if missing_vars:
                        print(f"[WARNING] Model does not contain all variables. Missing: {missing_vars}. Using full DataFrame for scoring.")
                        df_filtered = train_encoded[df_to_sl.columns]
                    else:
                        df_filtered = train_encoded[list(model_variables)]
                try:
                    score_bdeu = structure_score(model, df_filtered, scoring_method="bdeu")
                except Exception as e:
                    print(f"[ERROR] Could not compute BDeu score: {e}. Assigning NaN.")
                    score_bdeu = np.nan
                try:
                    score_bic = structure_score(model, df_filtered, scoring_method="bic-d")
                except Exception as e:
                    print(f"[ERROR] Could not compute BIC score: {e}. Assigning NaN.")
                    score_bic = np.nan
                print("Network quality BDeu:", score_bdeu)
                print("Network quality BIC:", score_bic)
                elapsed_time = time.time() - start_time
                key = f"{algorithm}_{score_method}_{sample_size}"
                trained_models[key] = model
                results.append({'Model': model,
                                'BDeu_Score': score_bdeu,
                                'BIC_Score': score_bic,
                                'Score_method': score_method,
                                'Algorithm': algorithm,
                                'Sample_Size': sample_size,
                                'Training_Time_Seconds': elapsed_time,
                                'Number_of_Edges': len(model.edges()),
                                'Number_of_df_variables': len(df_to_sl.columns)
                                })
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Failed: {algorithm} with {score_method}, sample_size={sample_size}: {error_msg}")
                errors.append({
                    'algorithm': algorithm,
                    'score_method': score_method,
                    'sample_size': sample_size,
                    'error': error_msg
                })
                continue
            
    results_structure_learning = pd.DataFrame(results)
    results_structure_learning = results_structure_learning.sort_values(by='BDeu_Score', ascending=False).reset_index(drop=True)
    
    print("\nStructure learning results:")
    print(results_structure_learning.to_string(index=False))
    results_structure_learning.to_csv(RESULTS_PATH, index=False)
    print(f"Results saved to: {RESULTS_PATH}")
    
    if not results_structure_learning.empty:
        # PC models are only valid when Number_of_Edges == Number_of_df_variables
        filtered_results = results_structure_learning.copy()
        for idx, row in filtered_results.iterrows():
            if row['Algorithm'] == 'pc' and row['Number_of_Edges'] != row['Number_of_df_variables']:
                print(f"[INFO] PC model with sample_size={row['Sample_Size']} discarded: Number_of_Edges ({row['Number_of_Edges']}) != Number_of_df_variables ({row['Number_of_df_variables']})")
                filtered_results = filtered_results.drop(idx)
        filtered_results = filtered_results.sort_values(by='BIC_Score', ascending=False).reset_index(drop=True)

        if not filtered_results.empty:
            best_row = filtered_results.iloc[0]
            best_model_key = f"{best_row['Algorithm']}_{best_row['Score_method']}_{int(best_row['Sample_Size'])}"
            best_score = best_row['BIC_Score']
            best_model_edges = len(trained_models[best_model_key].edges()) if best_model_key in trained_models and hasattr(trained_models[best_model_key], 'edges') else 'N/A'
            best_model = trained_models[best_model_key]
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            filename_best = f"./models/best_model_BIC_{best_model_key}_bicScore{best_score:.2f}_edges_{best_model_edges}_{timestamp_str}.pkl"
            with open(filename_best, 'wb') as f:
                pickle.dump(best_model, f)
            print(f"\nBest model (BIC) saved to: {filename_best}")

            if len(filtered_results) > 1:
                second_best_row = filtered_results.iloc[1]
                second_best_model_key = f"{second_best_row['Algorithm']}_{second_best_row['Score_method']}_{int(second_best_row['Sample_Size'])}"
                second_best_score = second_best_row['BIC_Score']
                second_best_model_edges = len(trained_models[second_best_model_key].edges()) if second_best_model_key in trained_models and hasattr(trained_models[second_best_model_key], 'edges') else 'N/A'
                if second_best_model_key in trained_models:
                    second_best_model = trained_models[second_best_model_key]
                    filename_second_best = f"./models/second_best_model_BIC_{second_best_model_key}_bicScore{second_best_score:.2f}_edges_{second_best_model_edges}_{timestamp_str}.pkl"
                    with open(filename_second_best, 'wb') as f:
                        pickle.dump(second_best_model, f)
                    print(f"Second best model (BIC) saved to: {filename_second_best}")
                else:
                    print("[WARNING] Second best model not found in trained_models.")
            else:
                print("No second best model to save.")
        else:
            print("[WARNING] No valid models after PC filtering. Cannot save best or second best model.")
            best_model = None
    else:
        print("[WARNING] No models were trained. Cannot save best or second best model.")
        best_model = None  
    
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(ERRORS_PATH, index=False)
        print(f"[INFO] Errors saved to '{ERRORS_PATH}'")
        
    try:
        if best_model:
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(best_model.nodes())
            nx_graph.add_edges_from(best_model.edges())
            pydot_graph = to_pydot(nx_graph)
            os.makedirs('./dag', exist_ok=True)
            pydot_graph.write_png(DAG_IMAGE_PATH)
            print(f'Best model image saved to {DAG_IMAGE_PATH}')
        else:
            print("[WARNING] No model trained, skipping image export.")
    except Exception as e:
        print(f"[ERROR] Could not save model image: {e}")
    
    
    if best_model:
        if TARGET_VARIABLE in best_model.nodes():
            markov_blanket = best_model.get_markov_blanket(TARGET_VARIABLE)
            print(f"Markov Blanket of '{TARGET_VARIABLE}':", markov_blanket)
        else:
            print(f"[WARNING] '{TARGET_VARIABLE}' not found in best model nodes.")
    else:
        print("[WARNING] No model trained, cannot compute Markov Blanket.")

if __name__ == "__main__":
    main()
