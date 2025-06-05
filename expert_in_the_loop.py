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
                    # Asegurarse de que las categorías en train_df sean un subconjunto de las definidas
                    # o que sean exactamente las mismas, antes de intentar la conversión.
                    # Esto es importante si train_df podría tener menos categorías que el df original
                    # desde el cual se generó dtype_definitions.
                    current_categories = train_df[col_name].unique()
                    # Convertir a string para comparación consistente, ya que JSON no guarda tipos numéricos exactos siempre
                    defined_categories_str = [str(cat) for cat in defs['categories']]
                    current_categories_str = [str(cat) for cat in current_categories if pd.notna(cat)]

                    # Verificar si todas las categorías actuales están en las definidas
                    # Esto es una heurística, podría necesitar ajustes más robustos
                    # si las categorías pueden diferir significativamente pero aún ser compatibles.
                    if all(cat_str in defined_categories_str for cat_str in current_categories_str):
                        cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                        train_df[col_name] = train_df[col_name].astype(cat_dtype)
                        # print(f"  Columna '{col_name}' convertida a CategoricalDtype (ordered={defs['ordered']}).")
                    else:
                        # print(f"[ADVERTENCIA] No se pudo convertir la columna {col_name} debido a categorías no coincidentes.")
                        # print(f"    Definidas: {defs['categories']}")
                        # print(f"    Actuales en train_df: {list(current_categories)}")
                        # Intentar la conversión de todas formas, pero podría fallar o dar resultados inesperados
                        # si las categorías no son compatibles.
                        try:
                            cat_dtype = pd.CategoricalDtype(categories=defs['categories'], ordered=defs['ordered'])
                            train_df[col_name] = train_df[col_name].astype(cat_dtype)
                            # print(f"  Columna '{col_name}' convertida (con advertencia de categorías) a CategoricalDtype (ordered={defs['ordered']}).")
                        except Exception as e_conv:
                            print(f"[ERROR] Fallo al convertir {col_name} incluso con categorías no coincidentes: {e_conv}")

                except Exception as e:
                    print(f"[ADVERTENCIA] No se pudo convertir la columna {col_name} al CategoricalDtype especificado: {e}")
                    # print(f"  Categorías esperadas: {defs['categories']}")
                    # print(f"  Categorías encontradas en los datos: {list(train_df[col_name].unique()) if hasattr(train_df[col_name], 'unique') else 'N/A'}")
        print("Tipos de datos después de la re-aplicación en train_df:")
        # train_df.info() # Descomentar para depuración si es necesario
    else:
        print(f"[ADVERTENCIA] No se encontró el archivo dtype_definitions.json en {dtype_definitions_path}. No se re-aplicarán los dtypes.")
    # --- Fin de la re-aplicación ---
    
    
    train_df.info()
    
    # 2. Aprendemos DAG con LLM
    descriptions = {
    "CONDICION": "Condición del caso de violencia reportado, como: nuevo, continuador, reincidente, reingreso.",
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
    "NIVEL_VIOLENCIA_DISTRITO":"Valoración del nivel de violencia en el distrito donde vive la victima",
    "SEGURO_VICTIMA":"¿Cuenta con algún tipo de seguro?",
    "TRATAMIENTO_VICTIMA":"¿Recibe actualmente algún tipo de tratamiento psicológico la victima?",
    "VINCULO_AFECTIVO":"'¿Tiene vínculos afectivos positivos la victima?",
    "VIOLENCIA_ECONOMICA": "¿La victima ha sufrido violencia económica?",
    "VIOLENCIA_PSICOLOGICA": "¿La victima ha sufrido violencia psicológica?",
    "VIOLENCIA_FISICA": "¿La victima ha sufrido violencia física?",
    "VIOLENCIA_SEXUAL": "¿La victima ha sufrido violencia sexual?",
    "HIJOS_VIVIENTES": "¿La victima tiene hijos vivientes?"
    }

    estimator = ExpertInLoop(train_df)
    dag = estimator.estimate(pval_threshold=0.03,
                            effect_size_threshold=0.0001, #Acepta relaciones mas debiles
                            variable_descriptions=descriptions,
                            use_llm=True,
                            llm_model="gemini/gemini-1.5-flash")
    
    #Guardar el DAG aprendido
    filename = "./models/dag_aprendido_with_llm.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(dag, f)
    
    print(f"\nEl DAG aprendido ha sido guardado en: {filename}")
    
    #Evaluación del DAG aprendido
    print("\nEvaluando el DAG aprendido...")
    
        
if __name__ == "__main__":
    main()
