import pickle
import os
import json
import time
from pgmpy.estimators import ExpertInLoop
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
import numpy as np
from pgmpy.metrics import structure_score

# --- Paths ---
TRAIN_ENCODED_PATH = './datasets/train_encoded.csv'
TRAIN_DF_PATH      = './datasets/train_df.csv'
VAL_ENCODED_PATH   = './datasets/val_encoded.csv'
VAL_DF_PATH        = './datasets/val_df.csv'
DTYPE_DEFS_PATH    = './uploads/dtype_definitions.json'
RESULTS_PATH       = './results/resultados_expert_in_the_loop.csv'

# --- Experiment configuration ---
LLM_MODELS = ['gemini/gemini-2.0-flash']
EXPERIMENTS = [
    {'effect_size_threshold': 0.0001, 'pval_threshold': 0.05},
    {'effect_size_threshold': 0.20,   'pval_threshold': 0.05},
]


def main():

    train_df_encoded = pd.read_csv(TRAIN_ENCODED_PATH)
    train_df         = pd.read_csv(TRAIN_DF_PATH)
    val_encoded      = pd.read_csv(VAL_ENCODED_PATH)
    val_df           = pd.read_csv(VAL_DF_PATH)

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
                    print(f"[WARNING] Could not convert column '{col_name}' to CategoricalDtype: {e}")
                    print(f"  Expected categories: {defs['categories']}")
                    print(f"  Categories found in data: {list(train_df[col_name].unique()) if hasattr(train_df[col_name], 'unique') else 'N/A'}")

    
    train_df_for_eil = train_df.dropna()
    dropped_rows = len(train_df) - len(train_df_for_eil)
    if dropped_rows > 0:
        print(f"[INFO] Dropped {dropped_rows} rows with NaNs before ExpertInLoop.")

    
    # Variable descriptions passed to the LLM (in Spanish, matching the original dataset domain)
    descriptions = {
    "CONDICION": "Condición del caso de violencia reportado, como: nuevo, reincidente (Cuando el acto ocurre nuevamente por el mismo agresor), reingreso (Cuando el acto ocurre por una persona agresora diferente a la primera vez).",
    "EDAD_VICTIMA": "El grupo etario de la victima",
    "LENGUA_MATERNA_VICTIMA": "Idioma o lengua materna de la victima con el que aprendió a hablar en su niñez.",
    "ETNIA_VICTIMA": "Como se identifica la victima en términos de raza, etnia o cultura, basado en sus costumbres y antepasados.",
    "AREA_RESIDENCIA_DOMICILIO": "El área donde la victima reside, urbano o rural",
    "ESTADO_CIVIL_VICTIMA": "El estado civil de la victima",
    "NIVEL_EDUCATIVO_VICTIMA": "Nivel educativo alcanzado por la victima.",
    "TRABAJA_VICTIMA": "La victima cuenta con un trabajo remunerado u ocupación para generar ingresos propios.",
    "VINCULO_AGRESOR_VICTIMA": "Vínculo o relación entre la victima y el agresor, como pareja, familiar, etc.",
    "AGRESOR_VIVE_CASA_VICTIMA": "Actualmente la presunta persona agresora vive en la casa de la victima.",
    "EDAD_AGRESOR": "El grupo etario del presunto agresor",
    "SEXO_AGRESOR":"Sexo del presunto agresor",
    "NIVEL_EDUCATIVO_AGRESOR": "Nivel educativo alcanzado por la presunta persona agresora.",
    "TRABAJA_AGRESOR":"La presunta persona agresora cuenta con un trabajo remunerado u ocupación para generar ingresos propios.",
    "FRECUENCIA_AGREDE":"Frecuencia que es agredida la victima por parte del agresor, como: diario, semanal, mensual, etc.",
    "NIVEL_DE_RIESGO_VICTIMA":"Valoración del nivel de riesgo que presenta la victima",
    "ESTUDIA":"¿Actualmente la victima estudia en una I.E./Colegio, Instituto Superior, Universidad u otro?",
    "ESTADO_AGRESOR_U_A":"Estado de la presunta persona agresora en la última agresión",
    "ESTADO_AGRESOR_G":"Estado de la presunta persona agresora generalmente",
    "ESTADO_VICTIMA_U_A":"Estado de la victima en la última agresión",
    "ESTADO_VICTIMA_G":"Estado de la persona usuaria generalmente",
    "REDES_FAM_SOC":"¿La victima cuenta con redes familiares o sociales, como amigos, colegas, vecinos?",
    "NIVEL_VIOLENCIA_DISTRITO":"Valoración del nivel de violencia en el distrito donde vive la victima, es la clasificacion del ratio de casos de violencia reportados en el distrito con respecto a la población total del distrito.",
    "SEGURO_VICTIMA":"¿La victima cuenta con algún tipo de seguro?",
    "TRATAMIENTO_VICTIMA":"¿Recibe actualmente algún tipo de tratamiento psicológico la victima?",
    "VINCULO_AFECTIVO":"'¿Tiene vínculos afectivos positivos la victima?",
    "VIOLENCIA_ECONOMICA": "¿La victima ha sufrido violencia económica?",
    "VIOLENCIA_PSICOLOGICA": "¿La victima ha sufrido violencia psicológica?",
    "VIOLENCIA_FISICA": "¿La victima ha sufrido violencia física?",
    "VIOLENCIA_SEXUAL": "¿La victima ha sufrido violencia sexual?",
    "HIJOS_VIVIENTES": "¿La victima tiene hijos vivientes?"
    }

    estimator = ExpertInLoop(train_df_for_eil) 

    print("\nStarting DAG learning with LLM...")

    results = []
    for exp in EXPERIMENTS:
        effect_size_threshold = exp['effect_size_threshold']
        pval_threshold        = exp['pval_threshold']
        for llm_model in LLM_MODELS:
            print(f"\nTraining with LLM: {llm_model} | effect_size_threshold={effect_size_threshold} | pval_threshold={pval_threshold}")
            start_time = time.time()
            try:
                dag = estimator.estimate(pval_threshold=pval_threshold, 
                                        effect_size_threshold=effect_size_threshold,
                                        variable_descriptions=descriptions,
                                        use_llm=True,
                                        llm_model=llm_model)
                
                elapsed_time = time.time() - start_time
                
                # Save learned model
                bn_model = DiscreteBayesianNetwork()
                bn_model.add_nodes_from(train_df_for_eil.columns)
                bn_model.add_edges_from(dag.edges())
                
                # Compute structure score
                # Ensure all nodes are present in the model
                for col in train_df_for_eil.columns:
                    if col not in bn_model.nodes():
                        bn_model.add_node(col)
                
                try:
                    score_bdeu = structure_score(bn_model, train_df_for_eil, scoring_method="bdeu")
                except Exception as e:
                    print(f"[ERROR] BDeu score failed for {llm_model}: {e}. Assigning NaN.")
                    score_bdeu = np.nan
                try:
                    score_bic = structure_score(bn_model, train_df_for_eil, scoring_method="bic-d")
                except Exception as e:
                    print(f"[ERROR] BIC score failed for {llm_model}: {e}. Assigning NaN.")
                    score_bic = np.nan
                print(f"BDeu network quality ({llm_model}):", score_bdeu)
                print(f"BIC network quality ({llm_model}):", score_bic)

                results.append({
                    'Model': bn_model,
                    'BDeu_Score': score_bdeu,
                    'BIC_Score': score_bic,
                    'Score_method': 'BDeu',
                    'Algorithm': llm_model,
                    'Sample_Size': len(train_df_for_eil),
                    'Training_Time_Seconds': elapsed_time,
                    'Number_of_Edges': len(bn_model.edges()),
                    'Number_of_df_variables': len(train_df_for_eil.columns),
                    'Effect_Size_Threshold': effect_size_threshold,
                    'Pval_Threshold': pval_threshold
                })
                
                # Save the learned DAG individually
                safe_llm_name = llm_model.replace('/', '_').replace('-', '_')
                filename = f"./models/learned_dag_with_llm_{safe_llm_name}_bicScore{score_bic:.2f}_effect{effect_size_threshold}_pval{pval_threshold}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(bn_model, f)
                print(f"Learned DAG ({llm_model}) saved to: {filename}")

            except Exception as e:
                print(f"[ERROR] ExpertInLoop failed for {llm_model}: {e}")

    # Crear DataFrame de resultados
    results_eil = pd.DataFrame(results)
    if not results_eil.empty:
        results_eil = results_eil.sort_values(by='BDeu_Score', ascending=False).reset_index(drop=True)
        
        print("\nComparison table of ExpertInLoop results:")
        print(results_eil.to_string(index=False))
        
        comparison_file_path = RESULTS_PATH
        results_eil.to_csv(comparison_file_path, index=False)
        print(f"Resultados guardados en: {comparison_file_path}")
    else:
        print("No se obtuvieron resultados para guardar.")
    
    
    
        
if __name__ == "__main__":
    main()
