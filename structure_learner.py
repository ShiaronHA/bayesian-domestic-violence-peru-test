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
from pgmpy.metrics import structure_score
import numpy as np

# Aprender mejor estructura de un modelo bayesiano a partir de un DataFrame
# Comparar modelos con 3 tamaños de muestra diferentes.
# Usará solo el train...


def preprocess_data(filepath):
        df = pd.read_csv(filepath, delimiter=',')
        print("Forma inicial del DataFrame:", df.shape)
        df = df.dropna()
        # Categorización explícita de variables ordinales
        df.NIVEL_EDUCATIVO_VICTIMA = pd.Categorical(df.NIVEL_EDUCATIVO_VICTIMA,
            categories=[
                'SIN NIVEL/INICIAL/BASICA ESPECIAL',
                'PRIMARIA INCOMPLETA',
                'PRIMARIA COMPLETA',
                'SECUNDARIA INCOMPLETA',
                'SECUNDARIA COMPLETA',
                'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
                'SUPERIOR TECNICO/UNIVERSITARIO COMPLETO',
                'MAESTRIA / DOCTORADO'],
            ordered=True)
        df.NIVEL_EDUCATIVO_AGRESOR = pd.Categorical(df.NIVEL_EDUCATIVO_AGRESOR,
            categories=[
                'SIN NIVEL/INICIAL/BASICA ESPECIAL',
                'PRIMARIA INCOMPLETA',
                'PRIMARIA COMPLETA',
                'SECUNDARIA INCOMPLETA',
                'SECUNDARIA COMPLETA',
                'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
                'SUPERIOR TECNICO/UNIVERSITARIO COMPLETO',
                'MAESTRIA / DOCTORADO'],
            ordered=True)
        df.EDAD_VICTIMA = pd.Categorical(
            df.EDAD_VICTIMA,
            categories=[
                'PRIMERA INFANCIA',
                'INFANCIA',
                'ADOLESCENCIA',
                'JOVEN',
                'ADULTO JOVEN',
                'ADULTO',
                'ADULTO MAYOR'
            ],
            ordered=True
        )
        df.EDAD_AGRESOR = pd.Categorical(
            df.EDAD_AGRESOR,
            categories=[
                'INFANCIA',
                'ADOLESCENCIA',
                'JOVEN',
                'ADULTO JOVEN',
                'ADULTO',
                'ADULTO MAYOR'
            ],
            ordered=True
        )
        df.FRECUENCIA_AGREDE = pd.Categorical(
            df.FRECUENCIA_AGREDE,
            categories=['MENSUAL', 'QUINCENAL', 'SEMANAL', 'INTERMITENTE', 'DIARIO'],
            ordered=True
        )
        df.NIVEL_DE_RIESGO_VICTIMA = pd.Categorical(
            df.NIVEL_DE_RIESGO_VICTIMA,
            categories=['LEVE', 'MODERADO', 'SEVERO'],
            ordered=True
        )
        df.NIVEL_VIOLENCIA_DISTRITO = pd.Categorical(
            df.NIVEL_VIOLENCIA_DISTRITO,
            categories=['Bajo', 'Medio', 'Alto'],
            ordered=True
        )

        nominal_cols = [
            'CONDICION', 'ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA', 'AREA_RESIDENCIA_DOMICILIO',
            'ESTADO_CIVIL_VICTIMA', 'TRABAJA_VICTIMA', 'VINCULO_AGRESOR_VICTIMA',
            'AGRESOR_VIVE_CASA_VICTIMA', 'TRATAMIENTO_VICTIMA', 'SEXO_AGRESOR', 'ESTUDIA',
            'ESTADO_AGRESOR_U_A','TRABAJA_AGRESOR', 'ESTADO_AGRESOR_G', 'ESTADO_VICTIMA_U_A', 'ESTADO_VICTIMA_G',
            'REDES_FAM_SOC', 'SEGURO_VICTIMA', 'VINCULO_AFECTIVO', 'VIOLENCIA_ECONOMICA',
            'VIOLENCIA_PSICOLOGICA', 'VIOLENCIA_SEXUAL', 'VIOLENCIA_FISICA', 'HIJOS_VIVIENTES'
        ]
        
        for col in nominal_cols:
            df[col] = pd.Categorical(df[col], ordered=False)
            
        # Convertir variables categóricas a códigos numéricos
        df_encoded = df.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)
        category_mappings = {
            col: dict(enumerate(df[col].cat.categories))
            for col in df.select_dtypes(['category']).columns
        }
        return df_encoded, df, category_mappings
    
    # --- Main para experimentos completos ---

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
            print(f"\nAprendiendo con Hill Climbing usando {scoring_method}...")
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
            bn_model = DiscreteBayesianNetwork(model.edges())
        elif algorithm == 'GES': #Causal Discovery
            print(f"\nAprendiendo con GES...")
            est = GES(df)
            model = est.estimate(scoring_method=scoring_method)
            bn_model = DiscreteBayesianNetwork(model.edges())
        elif algorithm == 'pc':
            print(f"\nAprendiendo con PC...")
            if scoring_method == 'pillai':
                assert not df.isnull().values.any(), "DataFrame contains NaN values"
                assert np.isfinite(df.to_numpy()).all(), "DataFrame contains inf values"
                # Validación: si hay columnas con baja varianza, eliminarlas
                low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
                print(f"Low-variance columns: {low_variance_cols}")
                est = PC(df)
                # --- Expert knowledge for PC ---
                model = est.estimate(
                    ci_test='pillai',
                    max_cond_vars=5,
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False
                )
            elif scoring_method == 'chi_square':
                est = PC(df)
                model = est.estimate(
                    variant='parallel',
                    ci_test='chi_square',
                    return_type='pdag',
                    significance_level=0.01,
                    max_cond_vars=3,
                    expert_knowledge=expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else None,
                    enforce_expert_knowledge=enforce_expert_knowledge if (expert_knowledge and enforce_expert_knowledge) else False,
                    n_jobs=-1,
                    show_progress=True
                )
            bn_model = DiscreteBayesianNetwork(model.edges())
        elif algorithm == 'mmhc':
            print("\nAprendiendo con MMHC (Max-Min Hill Climbing)...")
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
            bn_model = DiscreteBayesianNetwork(model.edges())
        else:
            raise ValueError("Algoritmo no soportado.")
        print("Estructura aprendida:", bn_model.edges())
        return bn_model

def main():
    
    filepath = 'data/df_processed.csv'
    df_encoded, df, dict = preprocess_data(filepath)
    total_size = len(df)

    # Paso 1: recolectar todas las categorías mínimas necesarias
    df_minimum = collect_all_categories(df)
    min_indices = set(df_minimum['index'])  # guardamos los índices usados
    df_minimum = df_minimum.set_index('index')

    # Paso 2: determinar cuántas filas más se necesitan para completar el train
    target_train_size = total_size - 1000
    n_missing = target_train_size - len(df_minimum)

    # Paso 3: muestreo aleatorio de filas adicionales, sin repetir las usadas
    df_remaining = df.drop(index=min_indices)
    df_extra = df_remaining.sample(n=n_missing, random_state=42)

    # Paso 4: formar el set de entrenamiento completo
    train_df = pd.concat([df_minimum, df_extra])
    train_indices = train_df.index

    # Paso 5: obtener el set codificado para el train
    train_encoded = df_encoded.loc[train_indices].reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    # Paso 6: validation = el resto
    all_indices = set(range(total_size))
    valid_indices = list(all_indices - set(train_indices))
    val_df = df.loc[valid_indices].reset_index(drop=True)
    val_encoded = df_encoded.loc[valid_indices].reset_index(drop=True)

    print("Train shape:", train_encoded.shape)
    print("Validation shape:", val_encoded.shape)
    
    # Guardar los DataFrames de entrenamiento y validación
    train_encoded.to_csv('./data/train_encoded.csv', index=False)
    train_df.to_csv('./data/train_df.csv', index=False)
    val_encoded.to_csv('./data/val_encoded.csv', index=False)
    val_df.to_csv('./data/val_df.csv', index=False)

    algorithms_to_experiment = [
        ('hill_climb', 'bic-d'), 
        #('hill_climb', 'k2'),
        ('hill_climb', 'bdeu'),
        ('pc', 'pillai'),
        ('pc', 'chi_square'),
	    ('GES','bic-d'),
        ('GES', 'bic-cg')
        #('mmhc', 'bic'),
	    #('mmhc', 'bdeu')
    ]
    sample_sizes = [20000, 40000, 50000]  # Sample sizes to experiment wit
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

        for algorithm, score_method in algorithms_to_experiment:
            if algorithm == 'hill_climb' or algorithm == 'pc':
                df_to_sl=sample_data_encoded
            else:
                df_to_sl=sample_data
                
            # Validación: si es PC y sample_size == 50000, saltar este experimento    
            if algorithm == 'pc' and sample_size == 70000:
                print (f"[AVISO] se omite PC con sample_size=70000")
                continue
            # --- Expert knowledge: LENGUA_MATERNA_VICTIMA->ETNIA_VICTIMA permitido, ETNIA_VICTIMA->LENGUA_MATERNA_VICTIMA prohibido ---
            expert_knowledge = None
            enforce_expert_knowledge = False
            if algorithm == 'pc':
                expert_knowledge = ExpertKnowledge(
                    required_edges=[
                        ('ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA')
                    ],
                    forbidden_edges=[
                        ('LENGUA_MATERNA_VICTIMA', 'ETNIA_VICTIMA')
                    ]
                )
                enforce_expert_knowledge = True
            print(f"\nAprendiendo estructura con {algorithm} with sample size = {sample_size}...")
            start_time = time.time()
            model = learn_structure(
                df_to_sl,
                algorithm=algorithm,
                scoring_method=score_method,
                output_path=f'./models/model_structure_31_{algorithm}_{score_method}_{sample_size}.pkl',
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
    results_structure_learning = pd.DataFrame(results)
    results_structure_learning = results_structure_learning.sort_values(by='BDeu_Score', ascending=False).reset_index(drop=True)
    
    print("\nTabla comparativa de resultados:")
    print(results_structure_learning.to_string(index=False))
    comparison_file_path = os.path.join('./uploads', 'resultados.csv')
    results_structure_learning.to_csv(comparison_file_path, index=False)
    print(f"Resultados guardados en: {comparison_file_path}")
    best_row = results_structure_learning.sort_values(by='BDeu_Score', ascending=False).iloc[0]
    best_model_key = f"{best_row['Algorithm']}_{best_row['Score_method']}_{int(best_row['Sample_Size'])}"
    best_score = best_row['BDeu_Score']
    best_model = trained_models[best_model_key]
    filename = f"./models/mejor_modelo_{best_model_key}_bDeuScore{best_score:.2f}_edges{len(best_model.edges())}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nEl mejor modelo ha sido guardado en: {filename}")
    
    

    # Guardar los mapeos (dict)
    dict_file_path = os.path.join('./uploads', 'categorical_mappings.json')
    with open(dict_file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
    print(f"Mapeos de categorías guardados en: {dict_file_path}")
    
    # Guardar imagen del best_model en la carpeta dag
    try:
        # Convertir el modelo a un grafo de networkx
        nx_graph = best_model.to_networkx()
        # Crear el objeto pydot
        pydot_graph = to_pydot(nx_graph)
        # Guardar como imagen PNG
        pydot_graph.write_png('./dag/best_model.png')
        print('Imagen del best_model guardada en ./dag/best_model.png')
    except Exception as e:
        print(f'No se pudo guardar la imagen del modelo: {e}')
    
    
    #Markov
    target_variable = 'NIVEL_DE_RIESGO_VICTIMA'
    markov_blanket = best_model.get_markov_blanket(target_variable)
    print(f"Markov Blanket de '{target_variable}':", markov_blanket)
    
    
    # Inferencia exacta
    
    #Las evidencias son val_df_encoded, extrae solo las columnas que se obtienen en markov_blanket
    evidence = val_encoded[markov_blanket].iloc[0].to_dict()
    print("Evidencia para la inferencia:", evidence)
    
    
    #bayesian_inference(model_rb, evidence)

if __name__ == "__main__":
    main()