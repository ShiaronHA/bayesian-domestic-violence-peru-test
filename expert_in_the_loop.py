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
from pgmpy.estimators import ExpertInLoop
import pandas as pd
from pgmpy.metrics import structure_score
import numpy as np



def main():
    
    # 1. Leemos los DataFrames de entrenamiento y validación
    train_df_encoded = pd.read_csv('./data/train_df_encoded.csv')
    train_df = pd.read_csv('./data/train_df.csv')
    val_df_encoded = pd.read_csv('./data/val_df_encoded.csv')
    val_df = pd.read_csv('./data/val_df.csv')
    print("DataFrames cargados correctamente.")
    
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
    dag = estimator.estimate(pval_threshold=0.05,
                            effect_size_threshold=0.05,
                            variable_descriptions=descriptions,
                            use_llm=True,
                            llm_model="gemini/gemini-1.5-flash")
    #Guardar el DAG aprendido
    filename = "./models/dag_aprendido_with_llm.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(dag, f)
        
if __name__ == "__main__":
    main()
