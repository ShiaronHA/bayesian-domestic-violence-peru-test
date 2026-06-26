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
            
    # 1. Load training and validation DataFrames
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames loaded successfully.")
    print("train_df shape:", train_df.shape)
    print("train_encoded shape:", train_encoded.shape)
    print("val_df shape:", val_df.shape)
    print("val_encoded shape:", val_encoded.shape)
    
    # Re-apply categorical dtypes to train_df to ensure category order
    dtype_definitions_path = './uploads/dtype_definitions.json'
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
    
    algorithms_to_experiment = [
        ('hill_climb', 'bic-d'), 
        #('hill_climb', 'k2'),
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
        # Step 1: get minimum sample covering all categories
        df_sample_min = collect_all_categories(train_df)
        min_indices_sample = set(df_sample_min['index'])  # already selected indices
        df_sample_min = df_sample_min.set_index('index')

        if sample_size < len(df_sample_min):
            print(f"[WARNING] Sample size {sample_size} is smaller than the minimum required ({len(df_sample_min)}). Skipping.")
            continue

        # Step 2: fill remaining rows randomly up to desired sample size
        n_extra = sample_size - len(df_sample_min)
        df_remaining_sample = train_df.drop(index=min_indices_sample)
        df_sample_extra = df_remaining_sample.sample(n=n_extra, random_state=42)

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

        for algorithm, score_method in algorithms_to_experiment:
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
                expert_knowledge  = ExpertKnowledge(
                    required_edges=[
                        ('ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA')
                    ],
                    forbidden_edges=[
                    ]
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
    comparison_file_path = os.path.join('./results', 'resultados_rb_classic.csv')
    results_structure_learning.to_csv(comparison_file_path, index=False)
    print(f"Results saved to: {comparison_file_path}")
    
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
        best_model = None  # no models trained
    
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv('./uploads/structure_learning_classic_errors.csv', index=False)
        print("[INFO] Errors saved to './uploads/structure_learning_classic_errors.csv'")
        
    try:
        if best_model:
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(best_model.nodes())
            nx_graph.add_edges_from(best_model.edges())
            pydot_graph = to_pydot(nx_graph)
            os.makedirs('./dag', exist_ok=True)
            pydot_graph.write_png('./dag/best_model_rb_classic.png')
            print('Best model image saved to ./dag/best_model_rb_classic.png')
        else:
            print("[WARNING] No model trained, skipping image export.")
    except Exception as e:
        print(f"[ERROR] Could not save model image: {e}")
    
    
    if best_model:
        target_variable = 'NIVEL_DE_RIESGO_VICTIMA'
        if target_variable in best_model.nodes():
            markov_blanket = best_model.get_markov_blanket(target_variable)
            print(f"Markov Blanket of '{target_variable}':", markov_blanket)
        else:
            print(f"[WARNING] '{target_variable}' not found in best model nodes.")
    else:
        print("[WARNING] No model trained, cannot compute Markov Blanket.")

if __name__ == "__main__":
    main()
