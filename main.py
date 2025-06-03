import pandas as pd
import pickle
import os
import json
import time
from sklearn.model_selection import train_test_split # Added import
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
from pgmpy.estimators import ExpertKnowledge

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Usara el mejor modelo aprendido para inferencia
# Y usara el train y val para aprender los parametros del modelo

# --- Preprocesamiento de datos ---

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
    

def learn_with_random_forest(train, target_col, val):
    
    # Dividir el conjunto de entrenamiento en características (X) y etiquetas (y)
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]
    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma de X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    rf_model = RandomForestClassifier(random_state = 42)

    rf_model.fit(X_train, y_train)

    accuracy_rf = accuracy_score(y_val, rf_model.predict(X_val))
    print(f"Precisión del modelo Random Forest: {accuracy_rf}")

    conf_matrix = confusion_matrix(y_val, rf_model.predict(X_val))
    print(f'Matriz de Confusión:\n{conf_matrix}')

    # Visualizamos la matriz de confusión con seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()
    
    #Guardar el grafico de la matriz de confusión
    conf_matrix_file_path = os.path.join('./plots', 'confusion_matrix_rf.png')
    plt.savefig(conf_matrix_file_path)
    print(f"Matriz de confusión guardada en: {conf_matrix_file_path}")

    # Reporte de clasificación
    class_report = classification_report(y_val, rf_model.predict(X_val))
    print(f'Classification Report:\n{class_report}')
    
    # Guarda el reporte de clasificación en un archivo
    report_file_path = os.path.join('./uploads', 'classification_report_rf.txt')
    with open(report_file_path, 'w') as f:
        f.write(class_report)
    print(f"Reporte de clasificación guardado en: {report_file_path}")
    
    return rf_model

def main():
    
    filepath = 'data/df_processed.csv'

    
    # 1. Leemos los DataFrames de entrenamiento y validación
    train_df_encoded = pd.read_csv('./data/train_df_encoded.csv')
    train_df = pd.read_csv('./data/train_df.csv')
    val_df_encoded = pd.read_csv('./data/val_df_encoded.csv')
    val_df = pd.read_csv('./data/val_df.csv')
    print("DataFrames cargados correctamente.")
    

    # 2. Cargamos el mejor modelo aprendido en structure_learner.py


    filename = f"./models/mejor_modelo_{best_model_key}_bDeuScore{best_score:.2f}_edges{len(best_model.edges())}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nEl mejor modelo ha sido guardado en: {filename}")
    
    
    
    # Aprendizaje de parámetros usando el training set
    model_rb = parameter_learning(best_model, train_df_encoded)
    
    #Experimento 2: Aprendiendo con Random Forest
    model_rf = learn_with_random_forest(train_df_encoded, 'NIVEL_DE_RIESGO_VICTIMA', val_df_encoded)
    
    #Evaluando modelo Random Forest
    
    
    #Evaluando modelo con Red Bayesiana
    print("\nEvaluando el modelo con Red Bayesiana...")
    
    #Markov
    target_variable = 'NIVEL_DE_RIESGO_VICTIMA'
    markov_blanket = best_model.get_markov_blanket(target_variable)
    print(f"Markov Blanket de '{target_variable}':", markov_blanket)
    
    
    # Inferencia exacta
    
    #Las evidencias son val_df_encoded, extrae solo las columnas que se obtienen en markov_blanket
    evidence = val_df_encoded[markov_blanket].iloc[0].to_dict()
    print("Evidencia para la inferencia:", evidence)
    
    
    # evidence = {
    #     'VINCULO_AGRESOR_VICTIMA': 1,
    #     'ESTUDIA': 1,
    #     'NIVEL_EDUCATIVO_VICTIMA': 2,
    #     'AREA_RESIDENCIA_DOMICILIO': 1
    # }
    
    #bayesian_inference(model_rb, evidence)

if __name__ == "__main__":
    main()
