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
from pgmpy.estimators import MmhcEstimator, BDeuScore


def main():
    print("Aprendiendo estructura de red bayesiana...")

    # Cargar datos
    df = pd.read_csv('data/df_processed.csv', delimiter=',')
    print("Forma inicial del DataFrame:", df.shape)

    # Convertir columnas de texto a categóricas
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype('category')

    # Preprocesar: eliminar nulos y codificar categorías
    print("Filas antes de dropna:", df.shape[0])
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype('category').cat.codes
    print("Filas después de dropna:", df.shape[0])

    # ===================
    # Hill Climbing
    # ===================
    print("\nAprendiendo con Hill Climbing...")
    hc_est = HillClimbSearch(df)
    hc_model = hc_est.estimate(scoring_method=BicScore(df))
    model_hc = BayesianNetwork(hc_model.edges())

    with open('./uploads/model_structure_hillClimbing.pkl', 'wb') as f:
        pickle.dump(model_hc, f)
    print("Modelo Hill Climbing guardado en './uploads/model_structure_hillClimbing.pkl'")
    print("Estructura Hill Climbing:", model_hc.edges())

    # ===================
    # MMHC
    # ===================
    print("\nAprendiendo con MMHC (Max-Min Hill Climbing)...")
    mmhc = MmhcEstimator(df)

    # Parte 1: Esqueleto
    skeleton = mmhc.mmpc()
    print("Esqueleto MMHC:", skeleton.edges())

    # Parte 2: Orientación con Hill Climbing + White List del esqueleto
    hc = HillClimbSearch(df)
    model_mmhc = hc.estimate(
        tabu_length=10,
        white_list=skeleton.to_directed().edges(),
        scoring_method=BDeuScore(df)
    )
    bn_mmhc = BayesianNetwork(model_mmhc.edges())

    with open('./uploads/model_structure_mmhc.pkl', 'wb') as f:
        pickle.dump(bn_mmhc, f)
    print("Modelo MMHC guardado en './uploads/model_structure_mmhc.pkl'")
    print("Estructura MMHC:", bn_mmhc.edges())


if __name__ == "__main__":
    main()
