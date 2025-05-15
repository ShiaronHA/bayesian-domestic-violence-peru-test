import pandas as pd
import pickle
import os
import json
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, BDeuScore, MmhcEstimator
from pgmpy.inference import BeliefPropagation
from pgmpy.readwrite import BIFWriter
from pgmpy.estimators import BayesianEstimator
from pgmpy.metrics import structure_score


def preprocess_data(filepath):
    df = pd.read_csv(filepath, delimiter=',')
    print("Forma inicial del DataFrame:", df.shape)
    df = df.dropna()
    df = df.astype('category')
    df_encoded = df.apply(lambda col: col.cat.codes if col.dtype.name == 'category' else col)
    category_mappings = {
    	col: dict(enumerate(df[col].cat.categories))
        for col in df.select_dtypes(['category']).columns
    }
    print(df_encoded)		
    return df_encoded, category_mappings

def learn_structure(df, algorithm='hill_climb', sampling = None, scoring_method=None, output_path=None):

    if algorithm == 'hill_climb':
        print(f"\nAprendiendo con Hill Climbing usando {scoring_method}...")
        est = HillClimbSearch(df)

        if scoring_method == 'bic':
            model = est.estimate(scoring_method=BicScore(df))
        elif scoring_method == 'bdeu':
            model = est.estimate(scoring_method=BDeuScore(df), max_indegree=4, max_iter=int(1e4))
        elif scoring_method == 'k2':
            model = est.estimate(scoring_method='k2score', max_indegree=4, max_iter=int(1e4))
        else:
            raise ValueError("Scoring method no soportado para Hill Climbing.")

        bn_model = BayesianNetwork(model.edges())

    elif algorithm == 'mmhc':
        print("\nAprendiendo con MMHC (Max-Min Hill Climbing)...")
        print("\nGeneramos sampling")
        # Generar un muestreo aleatorio de la base de datos
        df_sampled = df.sample(n=sampling, random_state=42)

        mmhc = MmhcEstimator(df_sampled)
        skeleton = mmhc.mmpc()
        hc = HillClimbSearch(df_sampled)
        model = hc.estimate(
            tabu_length=5,
            white_list=skeleton.to_directed().edges(),
            scoring_method=BDeuScore(df_sampled),
            max_indegree=3
        )
        bn_model = BayesianNetwork(model.edges())
    else:
        raise ValueError("Algoritmo no soportado.")

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(bn_model, f)
        print(f"Modelo guardado en '{output_path}'")

    print("Estructura aprendida:", bn_model.edges())
    
    #Evaluacion de modelo : structure_score
    #A higher score represents a better fit
    # scoring = BicScore(df)
    # score_value = scoring.score(bn_model)
    # print("Calidad de red BIC Score:", score_value)

    score = structure_score(bn_model, df, scoring_method="bdeu")
    print("Calidad de red BDeu Score:", score)
    return bn_model, score


def main():
    df, dict = preprocess_data('data/df_processed.csv')

    algorithms_to_experiment = ['hill_climb', 'mmhc']
    scoring_methods = ['bdeu','bic'] #k2,bic
    sampling_size = 50000
    results = []
    trained_models = {}

    # Aprendizaje de la estructura
    for algorithm in algorithms_to_experiment:
        print(f"\nAprendiendo estructura con {algorithm}...")
        if algorithm == 'hill_climb':
            for scoring_method in scoring_methods:
                print(f"\nUsando {scoring_method} como método de puntuación...")
                model, score = learn_structure(df, algorithm='hill_climb', scoring_method=scoring_method,
                                        output_path=f'./uploads/model_structure_29_{algorithm}_{scoring_method}.pkl')
        elif algorithm == 'mmhc':
            model, score = learn_structure(df, algorithm='mmhc', sampling=sampling_size,
                                    output_path=f'./uploads/model_structure_29_{algorithm}.pkl')
        key = f"{algorithm}_{scoring_methods if algorithm =='hill_climb' else 'BDeu'}"
        trained_models[key] = model
        results.append({'Model': model, 'BDeu_Score': score})


    comparison_df = pd.DataFrame(results)
    print("\nTabla comparativa de resultados:") 
    print(comparison_df.to_string(index=False))

    # Guardar los resultados en un archivo CSV
    comparison_file_path = os.path.join('./uploads', 'comparacion_resultados.csv')
    comparison_df.to_csv(comparison_file_path, index=False)
    print(f"Resultados guardados en: {comparison_file_path}")

    # Guardar mejor modelo
    best_row = comparison_df.sort_values(by='BDeu_Score', ascending=False).iloc[0]
    best_model_key = best_row['Model']
    best_score = best_row['BDeu_Score']
    best_model = trained_models[best_model_key]

    filename = f"./uploads/mejor_modelo_{best_model_key}_bic{best_score:.2f}.pkl"
    # Guardar el modelo como archivo pickle
    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nEl mejor modelo ha sido guardado en: {filename}")

    #Evaluando y guardando el mejor modelo
    model = best_model #Probando....

    #BayesianEstimator
    print("\nEstimando parámetros con BayesianEstimator...") 
    estimator = BayesianEstimator(model, df)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')

    # Guardar el modelo en la carpeta uploads con el formato BIF
    model_file_path = os.path.join('./uploads', 'mi_modelo_red.bif')
    writer = BIFWriter(model)
    writer.write_bif(model_file_path)

    print(f"El modelo ha sido guardado en: {model_file_path}")
    
    # Guardar los mapeos (dict)
    dict_file_path = os.path.join('./uploads', 'categorical_mappings.json')
    with open(dict_file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
    print(f"Mapeos de categorías guardados en: {dict_file_path}")


    #Inferencia exacta
    print("\nRealizando inferencia exacta ...")
    
    belief_propagation = BeliefPropagation(model)
    print("Calibrate belief propagation ...")
    belief_propagation.calibrate()
    print("Realizando primera inferencia ...")
    # Definir las evidencias
    evidence = {
        'VINCULO_AGRESOR_VICTIMA': 1,
        'ESTUDIA': 1,
        'NIVEL_EDUCATIVO_VICTIMA': 2,
        'AREA_RESIDENCIA_DOMICILIO': 1
    }

    result = belief_propagation.map_query(variables=['NIVEL_DE_RIESGO_VICTIMA'],evidence=evidence)
    print("Resultado de inferencia NIVEL_DE_RIESGO_VICTIMA:", result)
    # Guardar resultados de inferencia
    result = {k: int(v) if hasattr(v, 'item') else v for k, v in result.items()}
    result_file_path = os.path.join('./uploads', 'inferencia_resultado.json')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Resultado de la inferencia guardado en: {result_file_path}")


if __name__ == "__main__":
    main()
