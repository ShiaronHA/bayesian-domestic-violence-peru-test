import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyreadstat
import unicodedata
import seaborn as sns

# Define the input and output paths based on the project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'input_data')
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots') # Define a directory for plots

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
            
            # Map values, handling potential missing keys by keeping original value
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
            
            # Add year and part columns for traceability if needed
            #df['ANIO_REGISTRO'] = file_info['year']
            #df['PARTE_REGISTRO'] = file_info['part']
            
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
    # Concatenate, being mindful of potentially different columns
    # Use outer join to keep all columns, then decide on handling NaNs or specific columns
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
    # Obtiene las columnas categóricas y la cantidad de valores únicos para cada una
    categorical_columns = df.select_dtypes(include=['category']).columns
    if categorical_columns.empty:
        print("No categorical columns found to plot.")
        return

    unique_value_counts = {col: df[col].nunique() for col in categorical_columns}

    # Ordenar unique_value_counts por valores (cantidad de valores únicos) en orden descendente
    # Convert to list of items for slicing, then take top_n
    sorted_items = sorted(unique_value_counts.items(), key=lambda item: item[1], reverse=True)
    
    # Select top_n items
    top_n_items = sorted_items[:top_n]
    
    if not top_n_items:
        print(f"No data to plot after selecting top {top_n}.")
        return

    top_n_unique_value_counts = dict(top_n_items)

    # Crear el gráfico de barras usando los datos ordenados
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

def main():
    """Main function to run the data preprocessing pipeline."""
    print("Starting data preprocessing...")
    
    # Create output and plots directories if they don't exist
    for dir_path in [OUTPUT_DATA_DIR, PLOTS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    combined_df = load_and_process_data(INPUT_DATA_DIR, DATA_FILES_INFO)

    if not combined_df.empty:
        # --- Placeholder for further data cleaning and feature selection ---
        # Example: Select relevant columns (adjust column names as per your actual data)
        # relevant_columns = ['COL1', 'COL2', 'VIOLENCIA_FISICA', 'VIOLENCIA_PSICOLOGICA', ...]
        # if all(col in combined_df.columns for col in relevant_columns):
        #     combined_df = combined_df[relevant_columns]
        # else:
        #     print("Warning: Not all relevant_columns found in the combined dataframe.")
        
        # Example: Handle missing values (very basic example)
        # combined_df.dropna(subset=['TARGET_VARIABLE'], inplace=True) # If you have a specific target
        # For other columns, you might fill with a placeholder or use more sophisticated imputation
        
        # --- End of placeholder ---
        print("Forma inicial del DataFrame:", combined_df.shape)
        
        #Print types of columns
        print("Tipos de columnas:")
        print(combined_df.dtypes)
        
        columns_to_convert = ['INFORMANTE', 'FORMA_INGRESO', 'LENGUA_MATERNA_VICTIMA','ETNIA_VICTIMA','NIVEL_EDUCATIVO_VICTIMA','OCUPACION_VICTIMA','AGRESOR_EXTRANJERO','VINCULO_PAREJA', 'VINCULO_FAMILIAR','SIN_VINCULO', 'NIVEL_EDUCATIVO_AGRESOR','OCUPACION_AGRESOR','FACTOR_VICTIMA_DISCAPACIDAD','NIVEL_DE_RIESGO_VICTIMA','LUGAR_TENTATIVA_DE_FEMINICIDIO','SITUACION_AGRESOR','TIPO_VIOLENCIA','MODALIDAD_TENTATIVA_DE_FEMINICIDIO','MOVIL_TENTATIVA_DE_FEMINICIDIO','MOVIL_TENTATIVA_DE_FEMINICIDIO','MODALIDADES_VCM','FACTOR_VICTIMA_ABUSO_CONSUMO_ALCOHOL','FACTOR_VICTIMA_CONSUME_DROGAS']
        for c in columns_to_convert:
            if c in combined_df.columns: # Check if column exists before trying to convert
                combined_df[c] = combined_df[c].astype('category')
            else:
                print(f"Warning: Column '{c}' not found in DataFrame, skipping conversion.")
        
        #Grafico
        plot_save_path = os.path.join(PLOTS_DIR, 'categorical_unique_counts_initial.png') # Updated path
        plot_categorical_unique_counts(combined_df, top_n=50, save_path=plot_save_path)
        
        output_file_path = os.path.join(OUTPUT_DATA_DIR, 'df_processed.csv')
        try:
            combined_df.to_csv(output_file_path, index=False)
            print(f"Successfully saved processed data to: {output_file_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")
    else:
        print("No data to save.")
        
    print("Data preprocessing finished.")

if __name__ == "__main__":
    main()