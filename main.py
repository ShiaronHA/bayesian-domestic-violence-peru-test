import pandas as pd
import pickle
import os
import json
import time
from pgmpy.models import BayesianNetwork, DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MmhcEstimator, PC, GES
from pgmpy.inference import BeliefPropagation
from pgmpy.readwrite import BIFWriter
from pgmpy.estimators import BayesianEstimator
from pgmpy.metrics import structure_score
from pgmpy.estimators import K2, BDeu, BIC, AIC
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx

# --- Preprocesamiento de datos ---
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
    # Variables nominales (no ordenadas)
    nominal_cols = [
        'CONDICION', 'ETNIA_VICTIMA', 'LENGUA_MATERNA_VICTIMA', 'AREA_RESIDENCIA_DOMICILIO',
        'ESTADO_CIVIL_VICTIMA', 'TRABAJA_VICTIMA', 'VINCULO_AGRESOR_VICTIMA',
        'AGRESOR_VIVE_CASA_VICTIMA', 'TRATAMIENTO_VICTIMA', 'SEXO_AGRESOR', 'ESTUDIA',
        'ESTADO_AGRESOR_U_A', 'ESTADO_AGRESOR_G', 'ESTADO_VICTIMA_U_A', 'ESTADO_VICTIMA_G',
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

def validate_numeric_encoding(df):
    non_numeric = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        print(f"Convirtiendo columnas no numéricas a códigos: {non_numeric}")
        for col in non_numeric:
            df[col] = pd.Categorical(df[col]).codes
    return df

# --- Aprendizaje de estructura ---
def learn_structure(df, algorithm='hill_climb', scoring_method=None, output_path=None):
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
        else:
            raise ValueError("Scoring method no soportado para Hill Climbing.")
        bn_model = DiscreteBayesianNetwork(model.edges())
    elif algorithm == 'GES': #Causal Discovery
        print(f"\nAprendiendo con GES...")
        est = GES(df)
        model = est.estimate(scoring_method='bic-cg')
        bn_model = DiscreteBayesianNetwork(model.edges())
    elif algorithm == 'pc':
        print(f"\nAprendiendo con PC...")
        if scoring_method == 'pillai':
            df = validate_numeric_encoding(df)
            est = PC(df)
            model = est.estimate(ci_test='pillai') 
        elif scoring_method == 'chi_square':
            est = PC(df)
            model = est.estimate(ci_test='chi_square', variant="stable", max_cond_vars=4, return_type='dag')
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

def parameter_learning(model, df):
    print("\nEstimando parámetros con BayesianEstimator...")
    estimator = BayesianEstimator(model, df)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
    # Guardar el modelo en la carpeta uploads con el formato BIF
    model_file_path = os.path.join('./models', 'mi_modelo_red.bif')
    writer = BIFWriter(model)
    writer.write_bif(model_file_path)
    print(f"El modelo ha sido guardado en: {model_file_path}")
    
    return model

# --- Inferencia exacta ---
def bayesian_inference(model, evidence):
    print("\nRealizando inferencia exacta ...")
    belief_propagation = BeliefPropagation(model)
    print("Calibrate belief propagation ...")
    belief_propagation.calibrate()
    print("Realizando primera inferencia ...")
    result = belief_propagation.map_query(variables=['NIVEL_DE_RIESGO_VICTIMA'], evidence=evidence)
    print("Resultado de inferencia NIVEL_DE_RIESGO_VICTIMA:", result)
    result = {k: int(v) if hasattr(v, 'item') else v for k, v in result.items()}
    result_file_path = os.path.join('./uploads', 'inferencia_resultado.json')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Resultado de la inferencia guardado en: {result_file_path}")
    
# --- Main para experimentos completos ---
def main():
    
    filepath = 'data/df_processed.csv'
    df_encoded, df, dict = preprocess_data(filepath)
    # algorithms_to_experiment = [
    #     ('hill_climb', 'bic'),
    #     ('hill_climb', 'k2'),
    #     ('hill_climb', 'bdeu'),
    #     ('pc', 'bdeu'),
    #     ('mmhc', 'bic'),
    #     ('mmhc', 'bdeu')
    # ]
    algorithms_to_experiment = [
        ('hill_climb', 'bic'),
        ('hill_climb', 'bdeu'),
        ('pc', 'pillai'),
        #('pc', 'chi_square'),
        #('GES', 'bic-cg')
	('mmhc', 'bdeu')
    ]
    sample_sizes = [20000, 30000]
    results = []
    trained_models = {}
    for sample_size in sample_sizes:
        sample_data_encoded = df_encoded.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sample_data = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        for algorithm, score_method in algorithms_to_experiment:
            if algorithm == 'hill_climb' or algorithm == 'pc':
                df=sample_data_encoded
            else:
                df=sample_data    
            print(f"\nAprendiendo estructura con {algorithm} with sample size = {sample_size}...")
            start_time = time.time()
            model = learn_structure(df, algorithm=algorithm, scoring_method=score_method,
                output_path=f'./models/model_structure_29_{algorithm}_{score_method if algorithm == "hill_climb" else "BDeu"}_{sample_size}.pkl')
            model_variables = set(var for edge in model.edges() for var in edge)
            df_filtered = df_encoded[list(model_variables)] #Por revisar que df elegir codificado o no
            score = structure_score(model, df_filtered, scoring_method="bdeu")
            print("Calidad de red BDeue:", score)
            elapsed_time = time.time() - start_time
            key = f"{algorithm}_{score_method if algorithm == 'hill_climb' else 'bdeu'}_{sample_size}"
            trained_models[key] = model
            results.append({'Model': model,
                            'BDeu_Score': score,
                            'Score_method': score_method if algorithm == 'hill_climb' else 'bdeu',
                            'Algorithm': algorithm,
                            'Sample_Size': sample_size,
                            'Training_Time_Seconds': elapsed_time,
                            'Number_of_Edges': len(model),
                            'Number_of_df_variables': len(df.columns)
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
    
    # Aprendizaje de parámetros
    model = parameter_learning(best_model, df_encoded)
    
    # Inferencia exacta
    evidence = {
        'VINCULO_AGRESOR_VICTIMA': 1,
        'ESTUDIA': 1,
        'NIVEL_EDUCATIVO_VICTIMA': 2,
        'AREA_RESIDENCIA_DOMICILIO': 1
    }
    #bayesian_inference(model, evidence)

if __name__ == "__main__":
    main()
