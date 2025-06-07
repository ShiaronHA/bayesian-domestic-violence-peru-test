import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyreadstat
import unicodedata
import seaborn as sns
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

for directory in ["models", "uploads", "datasets", "results", "plots", "data" ]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# Define the input and output paths based on the project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'input_data')
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, 'datasets')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots') 

# List of data files to process
DATA_FILES_INFO = [
    {'name': 'BD Registro de Casos del CEM - Año 2017.sav', 'year': 2017, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2018.sav', 'year': 2018, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2019.sav', 'year': 2019, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2020.sav', 'year': 2020, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2021 - 1.sav', 'year': 2021, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2021 - 2.sav', 'year': 2021, 'part': 2},
    {'name': 'BD Registro de Casos del CEM - Año 2022.sav', 'year': 2022, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2023 - 1.sav', 'year': 2023, 'part': 1},
    {'name': 'BD Registro de Casos del CEM - Año 2023 - 2.sav', 'year': 2023, 'part': 2},
    {'name': 'BD Registro de Casos del CEM - Año 2024.sav', 'year': 2024, 'part': 1}
]

# Normalizador de texto
def normalize_text(text):
    if not isinstance(text, str):
        return str(text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.strip().upper()

def decode_dataframe(df, meta):
    """Decodes a DataFrame using its metadata.

    Args:
        df: The DataFrame to decode.
        meta: The metadata object containing value labels.

    Returns:
        A decoded DataFrame.
    """
    df_decoded = df.copy()
    for col in df.columns:
        if col in meta.variable_value_labels:
            value_labels = meta.variable_value_labels[col]
            normalized_value_labels = {}
            for k, v_label in value_labels.items():
                try:
                    normalized_value_labels[k] = normalize_text(v_label)
                except Exception as e:
                    print(f"Warning: Could not process label for key {k} in column {col}: {e}")
            
            df_decoded[col] = df_decoded[col].apply(lambda x: normalized_value_labels.get(x, normalize_text(str(x)) if pd.notna(x) else x))
            df_decoded[col] = df_decoded[col].astype('category')
    return df_decoded

def load_and_process_data(input_dir, files_info):
    """Loads, decodes, and concatenates data from specified SAV files."""
    all_dfs_decoded = []
    for file_info in files_info:
        file_path = os.path.join(input_dir, file_info['name'])
        try:
            print(f"Processing file: {file_info['name']}...")
            df, meta = pyreadstat.read_sav(file_path)
            df_decoded = decode_dataframe(df, meta)
            all_dfs_decoded.append(df_decoded)
            print(f"Successfully processed and decoded: {file_info['name']}")
        except Exception as e:
            print(f"Error processing file {file_info['name']}: {e}")
            continue

    if not all_dfs_decoded:
        print("No dataframes were loaded. Exiting.")
        return pd.DataFrame()

    print("Concatenating all decoded dataframes...")
    try:
        combined_df = pd.concat(all_dfs_decoded, ignore_index=True, sort=False) 
        print("Concatenation successful.")
    except Exception as e:
        print(f"Error during concatenation: {e}")
        return pd.DataFrame()
    return combined_df

def plot_categorical_unique_counts(df, top_n=50, save_path=None):
    """Generates and displays a bar plot of the top N categorical columns by unique value counts,
       and optionally saves it to a file."""
    categorical_columns = df.select_dtypes(include=['category']).columns
    if categorical_columns.empty:
        print("No categorical columns found to plot.")
        return

    unique_value_counts = {col: df[col].nunique() for col in categorical_columns}
    sorted_items = sorted(unique_value_counts.items(), key=lambda item: item[1], reverse=True)
    top_n_items = sorted_items[:top_n]
    
    if not top_n_items:
        print(f"No data to plot after selecting top {top_n}.")
        return

    top_n_unique_value_counts = dict(top_n_items)
    plt.figure(figsize=(15, 8))
    sns.barplot(x=list(top_n_unique_value_counts.keys()), y=list(top_n_unique_value_counts.values()))
    plt.xticks(rotation=90)
    plt.xlabel(f"Top {top_n} Categorical Columns (by unique values)")
    plt.ylabel("Cantidad de valores únicos")
    plt.title(f"Top {top_n} Cantidad de valores únicos por columna categórica")
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    #plt.show() 

def replace_values(df, columns):
    """Reemplaza los valores 1.0, 1, 0  por SI, SI, NO respectivamente en las columnas especificadas.

    Args:
        df: El DataFrame a modificar.
        columns: Una lista de nombres de columnas a modificar.

    Returns:
        El DataFrame modificado.
    """
    for column in columns:
        df[column] = df[column].replace({1.0: 'SI', '1': 'SI', '0': 'NO'})
    return df

def update_values_column(df, column):

  if column == 'NIVEL_DE_RIESGO_VICTIMA':
      replace_dict = {'RIESGO SEVERO': 'SEVERO', 'RIESGO MODERADO': 'MODERADO', 'RIESGO LEVE': 'LEVE'}
  elif column == 'INFORMANTE':
      replace_dict = {'LA PERSONA USUARIA NO ES LA PERSONA INFORMANTE': 'NO', 'LA PERSONA USUARIA SI ES LA PERSONA INFORMANTE': 'SI'}
  elif column == 'FORMA_INGRESO':
      replace_dict = {'PNP':'DERIVADO POR LA PNP',
                      'FICHA NOTIFICACION DE CASO':'FICHA DE NOTIFICACION DE CASO',
                      'PERSONAL DEL CEM DERIVA EL CASO':'CENTRO EMERGENCIA MUJER (CEM)','FICHA DE DERIVACION DE CEM':'CENTRO EMERGENCIA MUJER (CEM)','FICHA DE CEM':'CENTRO EMERGENCIA MUJER (CEM)',
                      'FICHA DE EIU':'Equipo Itinerante de Urgencia (EIU)',
                      'DERIVADO POR LA UGEL O LA DRE':'OFICIO EMITIDO POR LA UGEL O LA DRE',
                      'MINISTERIO PUBLICO':'DERIVADO POR EL MINISTERIO PUBLICO',
                      'ACUDE DIRECTAMENTE AL SERVICIO':'PERSONA ACUDE DIRECTAMENTE AL SERVICIO',
                      'ACUDE AL SERVICIO POR OTRO MOTIVO':'PERSONA ACUDE DIRECTAMENTE AL SERVICIO POR OTRO MOTIVO O SOLICITA ORIENTACION POR OTRAS MATERIAS',
                      'DERIVADO POR ESTABLECIMIENTO DE SALUD':'FICHA DE NOTIFICACION DEL ESTABLECIMIENTO DE SALUD (EE.SS.)',
                      'FICHA DE DERIVACION DE LINEA 100':'FICHA DE DERIVACION DE LA LINEA 100','FICHA LINEA 100':'FICHA DE DERIVACION DE LA LINEA 100',
                      'FICHA DE DERIVACION DE CHAT 100':'FICHA DE DERIVACION DEL CHAT 100','FICHA CHAT 100':'FICHA DE DERIVACION DEL CHAT 100',
                      'ESTRATEGIA RURAL (ER)':'SERVICIO DE ATENCION RURAL (SAR)','ESTRATEGIA RURAL':'SERVICIO DE ATENCION RURAL (SAR)',
                      'FICHA DE NOTIFICACION DEL CEM':'CENTRO EMERGENCIA MUJER (CEM)',
                      'PODER JUDICIAL':'DERIVADO POR EL PODER JUDICIAL',
                      'SAU':'SERVICIO DE ATENCION URGENTE (SAU)',
                      'CAI':'CENTRO DE ATENCION INSTITUCIONAL (CAI)'
                      }
  elif column == 'LENGUA_MATERNA_VICTIMA':
      replace_dict = {'LENGUA EXTRANJERA':'OTRA LENGUA EXTRANJERA', 'INGLES':'OTRA LENGUA EXTRANJERA',
                      'LENGUA DE SENAS':'LENGUA DE SENAS PERUANAS',
                      'OTRA LENGUA NATIVA':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'OTRA LENGUA MATERNA':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'YINE':'OTRA LENGUA INDIGENA U ORIGINARIA','WAMPIS':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'MATSES':'OTRA LENGUA INDIGENA U ORIGINARIA','HARAKBUT':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'KAKINTE (CAQUINTE)':'OTRA LENGUA INDIGENA U ORIGINARIA', 'KAKATAIBO':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'SHIPIBO':'SHIPIBO - KONIBO',
                      'NO ESCUCHA Y/O NO HABLA':'NO ESCUCHA/O NI HABLA/O','PERSONA CON DISCAPACIDAD FISICA PARA HABLAR':'NO ESCUCHA/O NI HABLA/O',
                      'YAMINAHUA':'OTRA LENGUA INDIGENA U ORIGINARIA','CAPANAHUA':'OTRA LENGUA INDIGENA U ORIGINARIA','AMAHUACA':'OTRA LENGUA INDIGENA U ORIGINARIA','CASHINAHUA':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'SHARANAHUA':'OTRA LENGUA INDIGENA U ORIGINARIA','IQUITU':'OTRA LENGUA INDIGENA U ORIGINARIA','CAUQUI':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'YANESHA':'OTRA LENGUA INDIGENA U ORIGINARIA','TIKUNA (TICUNA)':'OTRA LENGUA INDIGENA U ORIGINARIA','NOMATSIGENGA':'OTRA LENGUA INDIGENA U ORIGINARIA','OCAINA':'OTRA LENGUA INDIGENA U ORIGINARIA','MURUI - MUINANI':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'JAQARU':'OTRA LENGUA INDIGENA U ORIGINARIA','NANTI':'OTRA LENGUA INDIGENA U ORIGINARIA','BORA':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'MATSIGENKA':'MATSIGENKA / MACHIGUENGA',
                      'AWAJUN':'AWAJUN / AGUARUNA',
                      'SHAWI':'SHAWI / CHAYAHUITA',
                      'KANDOZI CHAPRA':'OTRA LENGUA INDIGENA U ORIGINARIA','MADIJA (CULINA)':'OTRA LENGUA INDIGENA U ORIGINARIA','TAUSHIRO':'OTRA LENGUA INDIGENA U ORIGINARIA',
                      'ARABELA':'OTRA LENGUA INDIGENA U ORIGINARIA','URARINA':'OTRA LENGUA INDIGENA U ORIGINARIA'
                      }
  elif column == 'ETNIA_VICTIMA':
      replace_dict = {'NO SABE':'NO SABE/NO RESPONDE','NO ESPECIFICA':'NO SABE/NO RESPONDE',
                      'OTRA ETNIA':'OTRO',
                      'AYMARA':'AIMARA',
                      'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE':'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO',
                      'NEGRO, MORENO, ZAMBO, MULATO, AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO':'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO',
                      'NEGRO, MORENO, ZAMBO, MULATO/PUEBLO AFROPERUANO':'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO',
                      'NATIVO O INDIGENA DE LA AMAZONIA':'INDIGENA U ORIGINARIO DE LA AMAZONIA',
                      'POBLACION AFROPERUANA (NEGRO, MULATO, ZAMBO, AFROPERUANO)':'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO',
                      'PERTENECIENTE DE OTRO PUEBLO INDIGENA U ORIGINARIO':'PERTENECIENTE O PARTE DE OTRO PUEBLO INDIGENA U ORIGINARIO'
      }
  elif column == 'LUGAR_TENTATIVA_DE_FEMINICIDIO':
      replace_dict = {'CALLE-VIA PUBLICA':'CALLE - VIA PUBLICA',
                      'CALLE  - VIA PUBLICA':'CALLE - VIA PUBLICA',
                      'CASA DE PERSONA AGRESORA':'CASA DE AGRESOR'
      }
  elif column == 'MODALIDAD_TENTATIVA_DE_FEMINICIDIO':
      replace_dict = {'DISPARO CON ARMA DE FUEGO':'DISPARO POR PROYECTIL DE ARMA DE FUEGO (PAF)',
                      'AGRESION OBJETO CONTUNDENTE':'AGRESIONES CON OBJETOS CONTUNDENTES, DUROS O PESADOS'
      }
  elif column == 'TIPO_VIOLENCIA':
      replace_dict = {'VIOLENCIA ECONOMICA - PATRIMONIAL': 'ECONOMICA O PATRIMONIAL', 'VIOLENCIA ECONOMICA O PATRIMONIAL': 'ECONOMICA O PATRIMONIAL',
                      'VIOLENCIA PSICOLOGICA':'PSICOLOGICA',
                      'VIOLENCIA FISICA':'FISICA',
                      'VIOLENCIA SEXUAL':'SEXUAL'}
  else:
      return df


  if column in df.columns:
      existing_values = df[column].unique().tolist()
      valid_replacements = {k: v for k, v in replace_dict.items() if k in existing_values}
      if valid_replacements:
          df[column] = df[column].replace(valid_replacements)

  return df

def cluster_column(df, column_name, threshold=0.8):
    """Clusters values in a specified column based on similarity.

    Args:
        df: The DataFrame containing the column.
        column_name: The name of the column to cluster.
        threshold: The distance threshold for forming clusters.

    Returns:
        A pandas Series with cluster labels for the specified column.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    column = df[column_name]
    unique_values = column.dropna().unique()

    if len(unique_values) > 1:

      if pd.api.types.is_categorical_dtype(column):
          unique_values = column.dropna().unique()

          embeddings = model.encode(unique_values.astype(str))
          print(f'La columa {column_name}: {unique_values}')
          similarities = model.similarity(embeddings, embeddings)
          distance_matrix = 1 - similarities
          linkage_matrix = linkage(distance_matrix, method='ward')
          clusters = fcluster(linkage_matrix, threshold, criterion='distance')

          # Creamos mapping
          cluster_mapping = dict(zip(unique_values, clusters))
          clusters = column.map(cluster_mapping)
          return pd.Series(clusters, index=column.index)
      else:
          return column
    else:
        print(f"La columna '{column_name}' tiene menos de 2 valores únicos. No se aplicará clustering.")
        return column


def update_ocupacion_columns(df):
    """Standardizes and clusters occupation columns."""
    occupation_cols = ['OCUPACION_VICTIMA', 'OCUPACION_AGRESOR']
    for col in occupation_cols:
        if col in df.columns:
            print(f"Processing occupation column: {col}")
            # Ensure the column is string type before normalize_text and clustering
            df[col] = df[col].astype(str).apply(lambda x: normalize_text(x) if pd.notna(x) else x)
            df[col +'_cluster'] = cluster_column(df, col)
        else:
            print(f"Warning: Occupation column '{col}' not found.")

    # Update original columns using cluster representatives only if cluster columns were created
    if 'OCUPACION_AGRESOR_cluster' in df.columns and 'OCUPACION_AGRESOR' in df.columns:
        cluster_representatives_agresor = df.groupby('OCUPACION_AGRESOR_cluster')['OCUPACION_AGRESOR'].first().to_dict()
        df['OCUPACION_AGRESOR'] = df['OCUPACION_AGRESOR_cluster'].map(cluster_representatives_agresor).astype('category')
    else:
        print("Warning: OCUPACION_AGRESOR_cluster or OCUPACION_AGRESOR not found, skipping update.")

    if 'OCUPACION_VICTIMA_cluster' in df.columns and 'OCUPACION_VICTIMA' in df.columns:
        cluster_representatives_victima = df.groupby('OCUPACION_VICTIMA_cluster')['OCUPACION_VICTIMA'].first().to_dict()
        df['OCUPACION_VICTIMA'] = df['OCUPACION_VICTIMA_cluster'].map(cluster_representatives_victima).astype('category')
    else:
        print("Warning: OCUPACION_VICTIMA_cluster or OCUPACION_VICTIMA not found, skipping update.")
            
    # Eliminar columnas que terminan en "_cluster"
    cols_to_drop = [col for col in df.columns if col.endswith('_cluster')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)        
    
    return df

def clean_metadata(df):
    """Cleans metadata, including clustering text fields like occupation."""
    print("Starting metadata cleaning...")
    
    columns_to_convert = ['INFORMANTE', 'FORMA_INGRESO', 'LENGUA_MATERNA_VICTIMA','ETNIA_VICTIMA','NIVEL_EDUCATIVO_VICTIMA','OCUPACION_VICTIMA','AGRESOR_EXTRANJERO','VINCULO_PAREJA', 'VINCULO_FAMILIAR','SIN_VINCULO', 'NIVEL_EDUCATIVO_AGRESOR','OCUPACION_AGRESOR','FACTOR_VICTIMA_DISCAPACIDAD','NIVEL_DE_RIESGO_VICTIMA','LUGAR_TENTATIVA_DE_FEMINICIDIO','SITUACION_AGRESOR','TIPO_VIOLENCIA','MODALIDAD_TENTATIVA_DE_FEMINICIDIO','MOVIL_TENTATIVA_DE_FEMINICIDIO','MOVIL_TENTATIVA_DE_FEMINICIDIO','MODALIDADES_VCM','FACTOR_VICTIMA_ABUSO_CONSUMO_ALCOHOL','FACTOR_VICTIMA_CONSUME_DROGAS']
    for c in columns_to_convert:
        df[c] = df[c].astype('category')
    
    # A. Columnas categoricas por corregir automáticamente (Valores igual o menor a 2)
    columns_to_modified = ['FACTOR_VICTIMA_DISCAPACIDAD', 'FACTOR_VICTIMA_ABUSO_CONSUMO_ALCOHOL', 'FACTOR_VICTIMA_CONSUME_DROGAS','AGRESOR_EXTRANJERO']
    df = replace_values(df, columns_to_modified)

    # B. Columnas categoricas por corregir manualmente
    for i in df.columns:
        df_cleaned = update_values_column(df, i)
    
    # C. Pre-proceso previo a la clusterización automática    
    df_cleaned = update_ocupacion_columns(df_cleaned)
            
    print("Metadata cleaning finished.")
    return df_cleaned

def clean_data_not_violence_and_mistakes(df):
    """Cleans data by removing cases not classified as domestic violence and other inconsistencies."""
    print("Starting specific data cleaning (not violence, mistakes)...")

    #A. Eliminar CONDICION == 'DERIVADO' OR 'CONTINUADOR', No cambia el fenómeno de violencia, ni agresor.
    df.drop(df[df['CONDICION'] == 'DERIVADO'].index, inplace=True)
    df.drop(df[df['CONDICION'] == 'CONTINUADOR'].index, inplace=True)
    
    #B. Eliminar registros donde  df['DPTO_DOMICILIO'] + df['PROV_DOMICILIO'] + df['DIST_DOMICILIO'] == 999999
    df = df[df['DPTO_DOMICILIO'] + df['PROV_DOMICILIO'] + df['DIST_DOMICILIO'] != '999999']
    
    return df

def clasificar_violencia_percentiles(ratio, series_ratio):
        if pd.isna(ratio): return 'Desconocido'
        
        p33 = series_ratio.quantile(0.33)
        p66 = series_ratio.quantile(0.66)
        
        if ratio <= p33: return 'Bajo'
        elif ratio <= p66: return 'Medio'
        else: return 'Alto'
        
def calculate_and_classify_violence_level(df, input_data_dir):
    """Calculates violence ratio and classifies violence level based on UBIGEO."""
    df_copy = df.copy()
    geo_cols = ['DPTO_DOMICILIO', 'PROV_DOMICILIO', 'DIST_DOMICILIO']

    print("Calculating and classifying violence level based on UBIGEO...")
    
    if not all(col in df_copy.columns for col in geo_cols):
        print("Warning: Geo columns for UBIGEO not found. Skipping violence level calculation.")
        return df 

    df_copy['UBIGEO'] = df_copy[geo_cols[0]].astype(str) + df_copy[geo_cols[1]].astype(str) + df_copy[geo_cols[2]].astype(str)
    df_copy['UBIGEO'] = df_copy['UBIGEO'].str.zfill(6)
    
    try:
        #df_ubigeo = pd.read_csv(os.path.join(input_data_dir, 'TB_UBIGEOS.csv'), delimiter=';')
        df_poblacion = pd.read_excel(os.path.join(input_data_dir, 'Poblacion_distritos.xlsx'))
    except FileNotFoundError as e:
        print(f"Error: UBIGEO or Population file not found: {e}. Skipping violence level calculation.")
        return df 

    df_poblacion['2022'] = pd.to_numeric(df_poblacion['2022'], errors='coerce')
    df_poblacion.dropna(subset=['2022'], inplace=True)
    df_poblacion['2022'] = df_poblacion['2022'].astype(int)

    # Contar casos por UBIGEO
    df_casos = df_copy.groupby('UBIGEO').size().reset_index(name='casos_reportados')
    df_poblacion_reducido = df_poblacion[['UBIGEO', '2022']].rename(columns={'2022': 'poblacion'})
    df_poblacion_reducido['UBIGEO'] = df_poblacion_reducido['UBIGEO'].astype(str).str.zfill(6)

    # Unir ambos DataFrames por UBIGEO
    df_merge = pd.merge(df_casos, df_poblacion_reducido, on='UBIGEO', how='left')
    df_merge['ratio_violencia'] = df_merge['casos_reportados'] / df_merge['poblacion']

       
    df_merge['NIVEL_VIOLENCIA_DISTRITO'] = df_merge['ratio_violencia'].apply(lambda x: clasificar_violencia_percentiles(x, df_merge['ratio_violencia']))
    df_merge['NIVEL_VIOLENCIA_DISTRITO'] = df_merge['NIVEL_VIOLENCIA_DISTRITO'].astype('category')
    
    df_copy = pd.merge(df_copy, df_merge[['UBIGEO', 'NIVEL_VIOLENCIA_DISTRITO']], on='UBIGEO', how='left')
    df_copy = df_copy.drop(columns=geo_cols + ['UBIGEO'], errors='ignore')
    return df_copy

def preprocess_data(df):
    
    df = clean_data_not_violence_and_mistakes(df)
    
    # Calculate and add violence level based on UBIGEO
    df = calculate_and_classify_violence_level(df, INPUT_DATA_DIR)

    # 1. Conversión de variables, agrupaciones
    
    ## A. Variable Seguro a variable binaria
    columns_seguro = ['PNP_SEGURO', 'PRIVADO_SEGURO', 'SIS_SEGURO', 'ESSALUD_SEGURO', 'OTRO_SEGURO']
    columns_seguro_no = ['NINGUN_SEGURO']
    
    for col in columns_seguro:
        df.loc[df[col] == 'SI', 'SEGURO_VICTIMA'] = 'SI'

    for col in columns_seguro_no:
        df.loc[df[col] == 'SI', 'SEGURO_VICTIMA'] = 'NO'
    
    df['SEGURO_VICTIMA'] = df['SEGURO_VICTIMA'].astype('category')
    df = df.drop(columns=columns_seguro + columns_seguro_no, errors='ignore')
    
    ## B.Victima-Agresor Peruana a binaria
    df.loc[df['VICTIMA_EXTRANJERA'] == 'SI', 'VICTIMA_PERUANA'] = 'NO'
    df.loc[df['AGRESOR_EXTRANJERO'] == 'SI', 'AGRESOR_PERUANO'] = 'NO'

    df = df.drop(columns=['VICTIMA_EXTRANJERA','AGRESOR_EXTRANJERO'])
    
    ## C. Victima recibe tratamiento a binaria
    cols_treatment = ['TRATAMIENTO_PSICOLOGICO','TRATAMIENTO_PSIQUIATRICO','ATENCION_MEDICA', 'OTRO_TRATAMIENTO','CONTINUA_RECIBIENDO_TRATAMIENTO']

    for col in cols_treatment:
        df.loc[df[col] == 'SI', 'TRATAMIENTO_VICTIMA'] = 'SI'

    df.loc[df['NINGUN_TRATAMIENTO'] == 'SI', 'TRATAMIENTO_VICTIMA'] = 'NO'
    df['TRATAMIENTO_VICTIMA'] = df['TRATAMIENTO_VICTIMA'].astype('category')
    df = df.drop(columns=cols_treatment + ['NINGUN_TRATAMIENTO'])
    
    ## D. Agregamos categoria NO a columnas con un solo valor (SI)
    cols_risk_factor_binary = [
        'FACTOR_AGRESOR_CONSUMO_ALCOHOL', 'FACTOR_AGRESOR_CONSUME_DROGA',
        'FACTOR_VICTIMA_DISCAPACIDAD', 'FACTOR_VICTIMA_ABUSO_CONSUMO_ALCOHOL', 
        'FACTOR_VICTIMA_CONSUME_DROGAS'
    ]
    for col in cols_risk_factor_binary:
        if col in df.columns:
            if df[col].dtype.name == 'category':
                if 'NO' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['NO'])
            df[col] = df[col].fillna('NO') 

    ## E. Conversión factor de riesgo (vínculo afectivo)
    cols_emotional_bond = ['VINCULO_AFECTIVO_FAMILIA','VINCULO_AFECTIVO_AMIGOS','VINCULO_AFECTIVO_VECINOS', 'VINCULO_AFECTIVO_ASOCIACIONES','VINCULO_AFECTIVO_ORGANIZACIONES_CIVICAS','VINCULO_AFECTIVO_COMPAÑEROS_TRABAJO','VINCULO_AFECTIVO_OTRO']
    cols_no_emotional_bond = ['VINCULO_AFECTIVO_NINGUNO']

    for col in cols_emotional_bond:
        df.loc[df[col] == 'SI', 'VINCULO_AFECTIVO'] = 'SI'

    for col in cols_no_emotional_bond:
        df.loc[df[col] == 'SI', 'VINCULO_AFECTIVO'] = 'NO'

    df['VINCULO_AFECTIVO'] = df['VINCULO_AFECTIVO'].astype('category')
    df = df.drop(columns=cols_emotional_bond + cols_no_emotional_bond)
    
    ## F. Conversión de signos de Tipo de violencia
    violence_types_map = {
        'VIOLENCIA_ECONOMICA': ['PERTURBACION_POSESION','MENOSCABO_TENENCIA_BIENES','PERDIDA_DERECHOS_PATRIMONIALES','LIMITACION_RECURSOS_ECONOMICOS','PRIVACION_MEDIOS_INDISPENSABLES','INCUMPLIMIENTO_OBLIGACION_ALIMENTARIA','CONTROL_DE_INGRESOS','PERCEPCION_SALARIO_MENOR','PROHIBIR_DES_LABORAL','SUSTRAER_INGRESOS', 'FRACCION_RECURSOS_NEC','OBLIGACION_ALIMENTOS','DESTRUIR_INST_TRABAJO','DESTRUIR_BIEN_PERSONAL','OTRA_VECON_PATRIM'],
        'VIOLENCIA_PSICOLOGICA': ['GRITOS_INSULTOS','VIOLENCIA_RACIAL','INDIFERENCIA','DISCR_ORIENTACION_SEXUAL','DISCR_GENERO','DISCR_IDENTIDAD_GENERO', 'RECHAZO','DESVALORIZACION_HUMILLACION','AMENAZA_QUITAR_HIJOS','OTRAS_AMENAZAS','PROHIBE_RECIBIR_VISITAS','PROHIBE_ESTUDIAR_TRABAJAR_SALIR', 'ROMPE_DESTRUYE_COSAS','VIGILANCIA_CONTINUA_PERSECUCION','BOTAR_CASA','AMENAZA_DE_MUERTE','ABANDONO','OTRA_VPSI'],
        'VIOLENCIA_FISICA': ['PUNTAPIES_PATADAS','PUÑETAZOS','BOFETADAS','JALONES_CABELLO','MORDEDURA','OTRAS_AGRESIONES','EMPUJONES','GOLPES_CON_PALOS', 'LATIGAZO','AHORCAMIENTO','HERIDAS_CON_ARMAS','GOLPES_CON_OBJETOS_CONTUNDENTES','NEGLIGENCIA','QUEMADURA','OTRA_VFIS'],
        'VIOLENCIA_SEXUAL': ['HOSTIGAMIENTO_SEXUAL','ACOSO_SEX_ESP_PUB','VIOLACION','TRATA_CON_FINES_EXPLOTACION_SEXUAL','EXPLOTACION_SEXUAL','PORNOGRAFIA', 'ACOSO_SEXUAL','OTRA_VSEX']
    }
    all_subtype_cols = []
    for main_type, subtypes in violence_types_map.items():
        existing_subtypes = [s for s in subtypes if s in df.columns]
        all_subtype_cols.extend(existing_subtypes)
        if existing_subtypes:
            df[main_type] = df[existing_subtypes].eq('SI').any(axis=1).map({True: 'SI', False: 'NO'})
        else:
            df[main_type] = 'NO' 
        df[main_type] = df[main_type].astype('category')
    df = df.drop(columns=all_subtype_cols, errors='ignore')
    
    ## G. Hijos a binaria
    if 'HIJAS_VIVAS' in df.columns and 'HIJOS_VIVOS' in df.columns:
        df['HIJOS_VIVIENTES'] = df.apply(lambda x: 'NO' if (x['HIJAS_VIVAS'] == 0.0 and x['HIJOS_VIVOS'] == 0.0) else 'SI', axis=1)
        df['HIJOS_VIVIENTES'] = df['HIJOS_VIVIENTES'].astype('category')
        df = df.drop(columns=['HIJAS_VIVAS','HIJOS_VIVOS'], errors='ignore')

    #2. Limpieza de duplicados
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"Cantidad de registros duplicados eliminados: {initial_rows - df.shape[0]}")
    
    #3. Limpieza de nulos
    umbral_nulos = 0.50 
    ratio_nulos = df.isnull().sum() / df.shape[0]
    columnas_a_eliminar_por_nulos = ratio_nulos[ratio_nulos > umbral_nulos].index
    df = df.drop(columns=columnas_a_eliminar_por_nulos)
    print(f"Columnas eliminadas por superar el umbral de nulos ({umbral_nulos*100}%): {len(columnas_a_eliminar_por_nulos)}")
    
    initial_rows_before_dropna = df.shape[0]
    df.dropna(inplace=True)
    print(f"Filas eliminadas por contener valores nulos restantes: {initial_rows_before_dropna - df.shape[0]}")
    print (f'Nro de registros completos con {df.shape[1]} variables: {df.shape[0]}')
    
    #4. Limpieza de outliers y ruido para EDAD_AGRESOR
    if 'EDAD_AGRESOR' in df.columns:
        print("Starting outlier treatment for EDAD_AGRESOR...")
        # Ensure 'EDAD_AGRESOR' is numeric. Original data might be float or object.
        # Coerce errors to NaN, which will be handled/dropped by outlier methods or explicitly.
        df['EDAD_AGRESOR'] = pd.to_numeric(df['EDAD_AGRESOR'], errors='coerce')
        
        # Store original 'EDAD_AGRESOR' (numeric and non-NaN) for plotting before outlier removal
        edad_agresor_original_for_plot = df['EDAD_AGRESOR'].dropna().copy()

        # Handle cases where the column might be all NaNs after coercion or mostly NaNs
        if edad_agresor_original_for_plot.empty:
            print("EDAD_AGRESOR column is empty or all NaN after numeric conversion. Skipping outlier treatment.")
        else:
            # --- IQR Method ---
            # Drop NaNs for IQR calculation if any survived or were introduced
            edad_agresor_iqr_data = df['EDAD_AGRESOR'].dropna()
            if not edad_agresor_iqr_data.empty:
                Q1 = edad_agresor_iqr_data.quantile(0.25)
                Q3 = edad_agresor_iqr_data.quantile(0.75)
                IQR_value = Q3 - Q1
                lower_bound_iqr = Q1 - 1.5 * IQR_value
                upper_bound_iqr = Q3 + 1.5 * IQR_value
                
                # Get indices from the original DataFrame where EDAD_AGRESOR is an outlier
                iqr_outliers_indices = df.index[
                    (df['EDAD_AGRESOR'] < lower_bound_iqr) | (df['EDAD_AGRESOR'] > upper_bound_iqr)
                ]
                print(f"IQR: Found {len(iqr_outliers_indices)} outliers.")
            else:
                iqr_outliers_indices = pd.Index([])
                print("IQR: No data for EDAD_AGRESOR after dropping NaNs.")

            # --- LOF Method ---
            # LOF requires a 2D array and no NaNs.
            df_lof_subset = df[['EDAD_AGRESOR']].dropna()
            if not df_lof_subset.empty and len(df_lof_subset) > 1 : # LOF needs more than 1 sample
                scaler = StandardScaler()
                # Ensure data is 2D for scaler
                scaled_edad_agresor = scaler.fit_transform(df_lof_subset[['EDAD_AGRESOR']])
                
                # Adjust n_neighbors if less than 20 samples are available
                n_neighbors_lof = min(20, len(df_lof_subset) - 1) if len(df_lof_subset) > 1 else 1
                if n_neighbors_lof > 0:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, contamination='auto')
                    lof_predictions = lof.fit_predict(scaled_edad_agresor)
                    
                    # LOF returns -1 for outliers. Get original indices.
                    lof_outliers_indices_relative = df_lof_subset[lof_predictions == -1].index
                    # The indices from df_lof_subset are already original df indices because we did not reset_index
                    lof_outliers_indices = lof_outliers_indices_relative
                    print(f"LOF: Found {len(lof_outliers_indices)} outliers using n_neighbors={n_neighbors_lof}.")
                else:
                    lof_outliers_indices = pd.Index([])
                    print("LOF: Not enough samples to apply LOF.")
            else:
                lof_outliers_indices = pd.Index([])
                print("LOF: No data or not enough samples for EDAD_AGRESOR after dropping NaNs.")

            # --- Find Common Outliers ---
            common_outliers_indices = iqr_outliers_indices.intersection(lof_outliers_indices)
            print(f"Found {len(common_outliers_indices)} common outliers using IQR and LOF.")

            # --- Plotting ---
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            if not edad_agresor_original_for_plot.empty:
                sns.boxplot(y=edad_agresor_original_for_plot)
                plt.title('EDAD_AGRESOR (Before Outlier Removal)')
            else:
                plt.title('EDAD_AGRESOR (Original - No Data)')
            plt.ylabel('Edad Agresor')

            if not common_outliers_indices.empty:
                # --- Save Removed Outliers Info ---
                removed_outliers_df = df.loc[common_outliers_indices, ['EDAD_AGRESOR']].copy()
                removed_outliers_df.index.name = 'original_index'
                # Use BASE_DIR for constructing path to 'uploads'
                removed_outliers_csv_path = os.path.join(BASE_DIR, 'uploads', 'EDAD_AGRESOR_outliers_removed.csv')
                removed_outliers_df.to_csv(removed_outliers_csv_path)
                print(f"Saved details of removed outliers to {removed_outliers_csv_path}")

                # Data after removing common outliers for the "after" plot
                edad_agresor_after_removal = df.drop(index=common_outliers_indices)['EDAD_AGRESOR'].dropna()
                
                plt.subplot(1, 2, 2)
                if not edad_agresor_after_removal.empty:
                    sns.boxplot(y=edad_agresor_after_removal)
                    plt.title('EDAD_AGRESOR (After Outlier Removal)')
                else:
                    plt.title('EDAD_AGRESOR (After - No Data Remaining)')
                plt.ylabel('Edad Agresor')
                
                # --- Remove Outliers from DataFrame ---
                df.drop(index=common_outliers_indices, inplace=True)
                print(f"Removed {len(common_outliers_indices)} common outliers from the DataFrame.")
            else:
                print("No common outliers to remove.")
                plt.subplot(1, 2, 2)
                if not edad_agresor_original_for_plot.empty: # Show original data again if no removal
                    sns.boxplot(y=edad_agresor_original_for_plot)
                    plt.title('EDAD_AGRESOR (No Outliers Removed)')
                else:
                    plt.title('EDAD_AGRESOR (No Outliers Removed - No Data)')
                plt.ylabel('Edad Agresor')
            
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, 'EDAD_AGRESOR_outliers.png')
            plt.savefig(plot_path)
            print(f"Box plot saved to {plot_path}")
            plt.close() # Close the plot to free memory
        
        print("Finished outlier treatment for EDAD_AGRESOR.")
    else:
        print("Column 'EDAD_AGRESOR' not found, skipping outlier treatment.")
    
    #4. Discretización de variables continuas
    ## Edad
    bins = [0, 6, 12, 18, 26, 36, 60, 121] 
    labels = ['PRIMERA INFANCIA', 'INFANCIA', 'ADOLESCENCIA', 'JOVEN', 'ADULTO JOVEN', 'ADULTO', 'ADULTO MAYOR']
    if 'EDAD_VICTIMA' in df.columns:
        df.loc[:, 'EDAD_VICTIMA'] = pd.cut(df['EDAD_VICTIMA'], bins=bins, labels=labels, right=False, ordered=True)
    if 'EDAD_AGRESOR' in df.columns:
        df.loc[:, 'EDAD_AGRESOR'] = pd.cut(df['EDAD_AGRESOR'], bins=bins, labels=labels, right=False, ordered=True)
    
    return df

def reduce_cardinality (df):
    
    # A. LENGUA_MATERNA_VICTIMA
    top2 = ['CASTELLANO', 'QUECHUA']
    # Reemplazar las clases que no están en top2 por 'OTRAS'
    df['LENGUA_MATERNA_VICTIMA'] = df['LENGUA_MATERNA_VICTIMA'].apply(
        lambda x: x if x in top2 else 'OTRAS'
    )
    df['LENGUA_MATERNA_VICTIMA'] = df['LENGUA_MATERNA_VICTIMA'].astype('category')
    
    # B. ETNIA_VICTIMA
    df['ETNIA_VICTIMA'] = df['ETNIA_VICTIMA'].astype(str)

    df['ETNIA_VICTIMA'] = df['ETNIA_VICTIMA'].replace({
        'AIMARA': 'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'BLANCO':'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'INDIGENA U ORIGINARIO DE LA AMAZONIA': 'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'PERTENECIENTE O PARTE DE OTRO PUEBLO INDIGENA U ORIGINARIO': 'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'NO SABE/NO RESPONDE': 'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'OTRO':'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO',
        'NEGRO, MORENO, ZAMBO, MULATO O AFRODESCENDIENTE O PARTE DEL PUEBLO AFROPERUANO':'AIMARA/BLANCO/INDIGENA/NEGRO/OTRO'
    })

    df['ETNIA_VICTIMA'] = df['ETNIA_VICTIMA'].astype('category')
    
    # C. NIVEL_EDUCATIVO_VICTIMA / NIVEL_EDUCATIVO_AGRESOR

    # Reemplazar las clases 'SIN NIVEL', 'INICIAL', 'BASICA ESPECIAL' por 'SIN NIVEL/INICIAL/BASICA ESPECIAL'
    df['NIVEL_EDUCATIVO_VICTIMA'] = df['NIVEL_EDUCATIVO_VICTIMA'].astype(str)

    df['NIVEL_EDUCATIVO_VICTIMA'] = df['NIVEL_EDUCATIVO_VICTIMA'].replace({
        'SIN NIVEL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'INICIAL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'BASICA ESPECIAL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'SUPERIOR TECNICO INCOMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
        'SUPERIOR UNIVERSITARIO INCOMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
        'SUPERIOR TECNICO COMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
        'SUPERIOR UNIVERSITARIO COMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
        'MAESTRIA / DOCTORADO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
    })
    df['NIVEL_EDUCATIVO_VICTIMA'] = df['NIVEL_EDUCATIVO_VICTIMA'].astype('category')
    
    df['NIVEL_EDUCATIVO_AGRESOR'].value_counts()

    # Reemplazar las clases 'SIN NIVEL', 'INICIAL', 'BASICA ESPECIAL' por 'SIN NIVEL/INICIAL/BASICA ESPECIAL'
    df['NIVEL_EDUCATIVO_AGRESOR'] = df['NIVEL_EDUCATIVO_AGRESOR'].astype(str)

    df['NIVEL_EDUCATIVO_AGRESOR'] = df['NIVEL_EDUCATIVO_AGRESOR'].replace({
        'SIN NIVEL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'INICIAL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'BASICA ESPECIAL': 'SIN NIVEL/INICIAL/BASICA ESPECIAL',
        'SUPERIOR TECNICO INCOMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
        'SUPERIOR UNIVERSITARIO INCOMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
        'SUPERIOR TECNICO COMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
        'SUPERIOR UNIVERSITARIO COMPLETO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
        'MAESTRIA / DOCTORADO':'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO',
    })

    df['NIVEL_EDUCATIVO_AGRESOR'] = df['NIVEL_EDUCATIVO_AGRESOR'].astype('category')
    
    # D. ESTADO_CIVIL_VICTIMA
    
    df['ESTADO_CIVIL_VICTIMA'] = df['ESTADO_CIVIL_VICTIMA'].astype(str)

    df['ESTADO_CIVIL_VICTIMA'] = df['ESTADO_CIVIL_VICTIMA'].replace({
        'VIUDO/A': 'VIUDO/DIVORCIADO/A',
        'DIVORCIADO/A':'VIUDO/DIVORCIADO/A'
    })

    df['ESTADO_CIVIL_VICTIMA'] = df['ESTADO_CIVIL_VICTIMA'].astype('category')

    # E. EDAD_VICTIMA 
    df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].astype(str)

    df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].replace({
        'INFANCIA': 'INFANCIA/ADULTO MAYOR',
        'PRIMERA INFANCIA': 'INFANCIA/ADULTO MAYOR',
        'ADULTO MAYOR': 'INFANCIA/ADULTO MAYOR'
    })

    df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].astype('category')
    
    # F. EDAD_AGRESOR 
    # df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].astype(str)

    # df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].replace({
    #     'INFANCIA': 'INFANCIA/ADULTO MAYOR',
    #     'PRIMERA INFANCIA': 'INFANCIA/ADULTO MAYOR',
    #     'ADULTO MAYOR': 'INFANCIA/ADULTO MAYOR'
    # })

    # df['EDAD_VICTIMA'] = df['EDAD_VICTIMA'].astype('category')
    
    
    return df

def filter_cardinality(df):
    # Obtener las columnas categóricas y la cantidad de valores únicos para cada una
    categorical_columns = df.select_dtypes(include=['category']).columns
    unique_value_counts = {col: df[col].nunique() for col in categorical_columns}
    high_cardinality_cols = [col for col, count in unique_value_counts.items() if count > 40]
    print(f'la variable a eliminar es: {high_cardinality_cols}')

    #A. Eliminamos variables con mucha cardinalidad (>50)
    df.drop(high_cardinality_cols, axis=1, inplace=True)
    
    #B. Reducción de cardinalidad en algunas variables
    df = reduce_cardinality(df)
    
    #C. Eliminamos variables con cardinalidad igual a 1.
    unique_cardinality_cols = [col for col, count in unique_value_counts.items() if count == 1]
    print(f'las variables a eliminar son: {unique_cardinality_cols}')
    df.drop(unique_cardinality_cols, axis=1, inplace=True)
    
    #D. Eliminamos variables binarias con una categoria menor al 3% de todos los datos
    
    binary_cols = [col for col, count in unique_value_counts.items() if count == 2]
    threshold = 0.03 * len(df)

    binary_cols_to_drop = []

    for col in binary_cols:
        value_counts = df[col].value_counts(dropna=False)
        if value_counts.min() < threshold:
            binary_cols_to_drop.append(col)

    print(f'Se eliminarán estas variables binarias por baja frecuencia: {binary_cols_to_drop}')
    df.drop(columns=binary_cols_to_drop, inplace=True)
    
    return df

def feature_selection(df):
    """Selects relevant features for analysis, dropping unnecessary columns."""
    print("Starting feature selection...")
    
    cols_to_exclude = ['FECHA_INGRESO', 'FORMA_INGRESO','CENTRO_POBLADO_DOMICILIO','CEM','TIPO_VIOLENCIA',
                   'INFORMANTE','DESEA_PATROCINIO_LEGAL','CUENTA_MEDIDAS_PROTECCION','FACTOR_AGRESOR_CONSUMO_ALCOHOL','FACTOR_AGRESOR_CONSUME_DROGA']

    df.drop(columns=cols_to_exclude, inplace=True)
    
    #Eliminamos variables con criterio de cardinalidad
    df_select = filter_cardinality(df)
    
    return df_select


def assign_dtypes(filepath):
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
                'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO'],
            ordered=True)
        df.NIVEL_EDUCATIVO_AGRESOR = pd.Categorical(df.NIVEL_EDUCATIVO_AGRESOR,
            categories=[
                'SIN NIVEL/INICIAL/BASICA ESPECIAL',
                'PRIMARIA INCOMPLETA',
                'PRIMARIA COMPLETA',
                'SECUNDARIA INCOMPLETA',
                'SECUNDARIA COMPLETA',
                'SUPERIOR TECNICO/UNIVERSITARIO INCOMPLETO',
                'SUPERIOR TECNICO/UNIVERSITARIO/POSTGRADO COMPLETO'],
            ordered=True)
        df.EDAD_VICTIMA = pd.Categorical(
            df.EDAD_VICTIMA,
            categories=[
                'INFANCIA/ADULTO MAYOR',
                'ADOLESCENCIA',
                'JOVEN',
                'ADULTO JOVEN',
                'ADULTO'
            ],
            ordered=True
        )
        df.EDAD_AGRESOR = pd.Categorical(
            df.EDAD_AGRESOR,
            categories=[
                #'INFANCIA',
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
        
        code_to_category_map = {
            col: dict(enumerate(df[col].cat.categories))
            for col in df.select_dtypes(['category']).columns
        }

        dtype_definitions = {}
        for col_name in df.select_dtypes(['category']).columns:
            cat_dtype = df[col_name].dtype
            if isinstance(cat_dtype, pd.CategoricalDtype):
                dtype_definitions[col_name] = {
                    'categories': list(cat_dtype.categories),
                    'ordered': cat_dtype.ordered
                }
        return df_encoded, df, code_to_category_map, dtype_definitions
    
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
    
def main():
    """Main function to run the data preprocessing pipeline."""
    print("Starting data preprocessing...")
    
    for dir_path in [OUTPUT_DATA_DIR, PLOTS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    combined_df = load_and_process_data(INPUT_DATA_DIR, DATA_FILES_INFO)

    if not combined_df.empty:
        print(f"Initial DataFrame shape: {combined_df.shape}")
        # print(f"Initial DataFrame columns: {combined_df.columns.tolist()}") # Optional: for verbose logging

        # 1. Clean metadata (e.g., cluster text fields like occupation)
        combined_df = clean_metadata(combined_df)
        print(f"DataFrame shape after metadata cleaning: {combined_df.shape}")
        # print(f"DataFrame columns after metadata cleaning: {combined_df.columns.tolist()}") # Optional

        # 2. Preprocess data (further cleaning, feature engineering, type conversions)
        combined_df = preprocess_data(combined_df)
        print(f"DataFrame shape after preprocessing: {combined_df.shape}")
        # print(f"DataFrame columns after preprocessing: {combined_df.columns.tolist()}") # Optional
        
        print("Final DataFrame types:")
        print(combined_df.dtypes.to_string()) # Use to_string() for better console output of many dtypes
        
        # Seleccion de características para el análisis
        combined_df = feature_selection(combined_df)
        print(f"DataFrame shape after feature selection: {combined_df.shape}")
        
        combined_df = combined_df.drop_duplicates()
        print(f"DataFrame shape after drop duplicates: {combined_df.shape}")
        
        #Eliminar filas con nulos restantes
        combined_df.dropna(inplace=True)
        
        # 3. Plot final categorical unique counts
        plot_save_path_final = os.path.join(PLOTS_DIR, 'categorical_unique_counts_final.png')
        plot_categorical_unique_counts(combined_df, top_n=50, save_path=plot_save_path_final)
        
        # 4. Save the fully processed DataFrame
        output_file_path = os.path.join(OUTPUT_DATA_DIR, 'df_full_processed.csv')
        try:
            combined_df.to_csv(output_file_path, index=False)
            print(f"Successfully saved fully processed data to: {output_file_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")
            
        # 5. Asignar dtypes a las columnas categóricas
        filepath = 'datasets/df_full_processed.csv'
        df_encoded, df, code_to_category_map, dtype_definitions = assign_dtypes(filepath)
        # 6. Dividir el DataFrame en dos partes: una para el análisis y otra para la predicción
        
        # Paso 0: cargar los datos
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
        train_encoded.to_csv('./datasets/train_encoded.csv', index=False)
        train_df.to_csv('./datasets/train_df.csv', index=False)
        val_encoded.to_csv('./datasets/val_encoded.csv', index=False)
        val_df.to_csv('./datasets/val_df.csv', index=False)
        
        # Guardar los mapeos de códigos de categorías (anteriormente 'dict')
        dict_file_path = os.path.join('./uploads', 'categorical_mappings.json')
        with open(dict_file_path, 'w', encoding='utf-8') as f:
            json.dump(code_to_category_map, f, indent=4, ensure_ascii=False)
        print(f"Mapeos de códigos de categorías guardados en: {dict_file_path}")

        # Guardar las definiciones de dtype
        dtype_definitions_path = os.path.join('./uploads', 'dtype_definitions.json')
        with open(dtype_definitions_path, 'w', encoding='utf-8') as f:
            json.dump(dtype_definitions, f, indent=4, ensure_ascii=False)
        print(f"Definiciones de Dtype guardadas en: {dtype_definitions_path}")
        
    else:
        print("No data to process or save.")
        
    print("Data preprocessing finished.")

if __name__ == "__main__":
    main()
