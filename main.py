from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle

from sklearn.metrics import f1_score
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork


def main():
    print("Aprendiendo estructura de red bayesiana con Hill Climbing...")

    # Cargar datos
    df = pd.read_csv('data/df_processed_45.csv')
    print(df.shape)
    # Convertir columnas de texto a categóricas
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category')

    print("Filas antes de dropna:", df.shape[0])
    # Preprocesar: eliminar nulos y codificar
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype('category').cat.codes
    
    print("Filas después de dropna:", df.shape[0])

    # Aprendizaje de estructura con Hill Climbing
    est = HillClimbSearch(df)
    best_model = est.estimate(scoring_method=BicScore(df))
    print("Mejor modelo encontrado por Hill Climbing:")
    print(best_model.edges())
    model_hillClimb = BayesianNetwork(best_model.edges())

    # Guardar estructura aprendida
    with open('./uploads/model_structure_hillClimbing.pkl', 'wb') as f:
        pickle.dump(model_hillClimb, f)

    print("Modelo estructural aprendido y guardado en './uploads/model_structure_hillClimbing.pkl'")


if __name__ == "__main__":
    main()
