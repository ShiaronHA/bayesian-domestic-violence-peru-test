from pgmpy.estimators import BayesianEstimator
import pickle
import os
import json
import time
from sklearn.model_selection import train_test_split # Added import
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from pgmpy.estimators import ExpertInLoop
from pgmpy.estimators.CITests import chi_square # Added import
import pandas as pd
import numpy as np


def main():

    # 1. Leemos los DataFrames de entrenamiento y validación
    train_df_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames cargados correctamente.")

    # --- Cargar dtype_definitions y re-aplicar a train_df ---
    dtype_definitions_path = './uploads/dtype_definitions.json'
    if os.path.exists(dtype_definitions_path):
        with open(dtype_definitions_path, 'r', encoding='utf-8') as f:
            dtype_definitions = json.load(f)
    
    print("\\nRe-aplicando dtypes categóricos a train_df...")
    
    for col_name, defs in dtype_definitions.items():
            if col_name in train_df.columns:
                try:
                    cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                    train_df[col_name] = train_df[col_name].astype(cat_dtype)
                except Exception as e:
                    print(f"[ADVERTENCIA] No se pudo convertir la columna {col_name} al CategoricalDtype especificado: {e}")
                    print(f"  Categorías esperadas: {defs['categories']}")
                    print(f"  Categorías encontradas en los datos: {list(train_df[col_name].unique()) if hasattr(train_df[col_name], 'unique') else 'N/A'}")

    
    train_df.info()
    
    # Attempt to resolve XGBoost ValueError by removing rows with any NaNs
    # This is a test to see if residual NaNs are causing .cat.codes to be -1
    original_rows = len(train_df)

    #Imprime cuantos NaNs hay en cada columna
    print("\n[INFO] Conteo de NaNs en cada columna del DataFrame de entrenamiento:")
    print(train_df.isna().sum())
    # Elimina filas con NaNs
    train_df_for_eil = train_df.dropna()
    dropped_rows = original_rows - len(train_df_for_eil)
    
    if dropped_rows > 0:
        print(f"[INFO] Dropped {dropped_rows} rows from train_df due to NaNs before passing to ExpertInLoop.")

    
    # 2. Aprendemos DAG con LLM
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
    "REDES_FAM_SOC":"¿Cuenta con redes familiares o sociales?",
    "NIVEL_VIOLENCIA_DISTRITO":"Valoración del nivel de violencia en el distrito donde vive la victima, es la clasificacion del ratio de casos de violencia reportados en el distrito con respecto a la población total del distrito.",
    "SEGURO_VICTIMA":"¿Cuenta con algún tipo de seguro?",
    "TRATAMIENTO_VICTIMA":"¿Recibe actualmente algún tipo de tratamiento psicológico la victima?",
    "VINCULO_AFECTIVO":"'¿Tiene vínculos afectivos positivos la victima?",
    "VIOLENCIA_ECONOMICA": "¿La victima ha sufrido violencia económica?",
    "VIOLENCIA_PSICOLOGICA": "¿La victima ha sufrido violencia psicológica?",
    "VIOLENCIA_FISICA": "¿La victima ha sufrido violencia física?",
    "VIOLENCIA_SEXUAL": "¿La victima ha sufrido violencia sexual?",
    "HIJOS_VIVIENTES": "¿La victima tiene hijos vivientes?"
    }

    estimator = ExpertInLoop(train_df_for_eil) # Use the NaN-dropped version

    # # --- Add logging to identify candidate edges for LLM orientation ---
    # print("\\nBuilding skeleton for inspection to identify LLM candidates...")
    # try:
    #     # Assuming 'chi_square' is the default test for discrete data in ExpertInLoop
    #     # and using the same pval_threshold as in the estimate call.
    #     current_pval_threshold = 0.03 # Match the pval_threshold in estimate()
        
    #     # ExpertInLoop internally uses CITests.get_instance, which for discrete data and default
    #     # 'test' parameter would be ChiSquare.
    #     # The max_cond_vars and max_combinations defaults in ExpertInLoop.estimate are 100 and None.
    #     ci_estimator = chi_square(train_df_for_eil)
    #     skeleton, _ = ci_estimator.build_skeleton_from_data(
    #         significance_level=current_pval_threshold,
    #         max_cond_vars=100,      # Default from ExpertInLoop.estimate
    #         max_combinations=None   # Default from ExpertInLoop.estimate
    #     )
    #     print("Edges in the statistically derived skeleton (potential candidates for LLM orientation):")
    #     candidate_edges = list(skeleton.edges())
    #     if not candidate_edges:
    #         print("  No edges found in the initial skeleton.")
    #     else:
    #         for u, v in candidate_edges:
    #             print(f"  - ({u}, {v})")
    #     print("--- End of candidate edge list ---\\n")
    # except Exception as e:
    #     print(f"[ERROR] Could not build skeleton for inspection: {e}")
    # # --- End of added logging ---

    dag = estimator.estimate(pval_threshold=0.03, #0.05
                            effect_size_threshold=0.001, #0.0001
                            variable_descriptions=descriptions,
                            use_llm=True,
                            llm_model="gemini/gemini-1.5-flash") #gemini-pro, gpt-4
    
    #Guardar el DAG aprendido
    filename = "./models/dag_aprendido_with_llm.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(dag, f)
    
    print(f"\nEl DAG aprendido ha sido guardado en: {filename}")
    
    #Evaluación del DAG aprendido
    print("\nEvaluando el DAG aprendido...")
    
    
    
        
if __name__ == "__main__":
    main()
