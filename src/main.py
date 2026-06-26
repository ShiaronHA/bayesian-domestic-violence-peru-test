import os
import json
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import GibbsSampling
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx


def parameter_learning(model, df):
    """
    Estimate parameters for a Bayesian Network using BayesianEstimator.
    """
    print("\nEstimating parameters with BayesianEstimator...")
    estimator = BayesianEstimator(model, df)
    model.fit(df, BayesianEstimator, prior_type='BDeu')
    return model


def bayesian_inference_exact(model, evidences_df, variable_name, model_name):
    """
    Perform exact inference using Belief Propagation for multiple cases.
    Results are saved after each case for robustness.
    """
    result_file_path = os.path.join('./results', f'inference_exact_rb_ypred_batch_{model_name}.json')
    calibration_error_file_path = os.path.join('./results', f'inference_rb_error_ypred_batch_{model_name}.json')

    print("\nPerforming exact inference for multiple cases...")
    belief_propagation = BeliefPropagation(model)
    print("Calibrating Belief Propagation...")
    try:
        belief_propagation.calibrate()
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        error_result = {"error": "Belief Propagation calibration failed", "details": str(e)}
        with open(calibration_error_file_path, 'w', encoding='utf-8') as f:
            json.dump([error_result], f, indent=4, ensure_ascii=False)
        print(f"Calibration error saved to: {calibration_error_file_path}")
        return [error_result]

    all_results = []
    start_case_index = 0

    # Load previous results if available
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, list) and all(isinstance(item, dict) for item in loaded_data):
                is_calibration_error = False
                if len(loaded_data) == 1 and "error" in loaded_data[0]:
                    if "Belief Propagation calibration failed" in loaded_data[0]["error"]:
                        is_calibration_error = True
                if not is_calibration_error:
                    all_results = loaded_data
                    start_case_index = len(all_results)
                    print(f"Loaded {start_case_index} previous results from '{result_file_path}'.")
                else:
                    print(f"File '{result_file_path}' contained a calibration error. Restarting inference.")
            else:
                print(f"[WARNING] File '{result_file_path}' does not contain a valid results list. Restarting.")
        except json.JSONDecodeError:
            print(f"[WARNING] JSON decode error in '{result_file_path}'. Restarting.")
        except Exception as e:
            print(f"[WARNING] Could not load previous results from '{result_file_path}': {e}. Restarting.")

    total_max_predictions = 100
    batch_size = 7
    num_total_to_process = min(total_max_predictions, evidences_df.shape[0])

    if start_case_index >= num_total_to_process:
        print(f"All {num_total_to_process} cases already processed according to '{result_file_path}'.")
        return all_results

    print(f"Performing inference for {num_total_to_process} cases, in batches of {batch_size}.")
    if start_case_index > 0:
        print(f"Resuming from case {start_case_index + 1}.")

    for batch_start_idx in range(0, num_total_to_process, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, num_total_to_process)
        if batch_end_idx <= start_case_index:
            print(f"  Batch {batch_start_idx + 1} to {batch_end_idx} already processed. Skipping.")
            continue
        current_batch_df = evidences_df.iloc[batch_start_idx:batch_end_idx]
        if current_batch_df.empty:
            continue
        print(f"  Processing batch: cases {batch_start_idx + 1} to {batch_end_idx} (of {num_total_to_process})...")
        for i in range(current_batch_df.shape[0]):
            actual_case_index_in_original_df = batch_start_idx + i
            if actual_case_index_in_original_df < start_case_index:
                continue
            evidence_dict = current_batch_df.iloc[i].to_dict()
            serializable_evidence_dict = {}
            for k, v in evidence_dict.items():
                if isinstance(v, float) and v.is_integer():
                    serializable_evidence_dict[k] = int(v)
                elif isinstance(v, (np.integer, np.int64)):
                    serializable_evidence_dict[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    serializable_evidence_dict[k] = float(v)
                elif pd.isna(v):
                    serializable_evidence_dict[k] = None
                else:
                    serializable_evidence_dict[k] = v
            print(f"    Case {actual_case_index_in_original_df + 1}/{num_total_to_process} - Evidence: {serializable_evidence_dict}")
            try:
                result = belief_propagation.map_query(variables=[variable_name], evidence=evidence_dict)
                processed_result = {k_res: int(v_res) if hasattr(v_res, 'item') else v_res for k_res, v_res in result.items()}
                all_results.append(processed_result)
            except Exception as e:
                error_message = f"Error processing case {actual_case_index_in_original_df + 1}"
                print(f"    [ERROR] {error_message} with evidence {serializable_evidence_dict}: {e}")
                all_results.append({"error": str(e), "details": error_message,
                                    "evidence_case_number": actual_case_index_in_original_df + 1,
                                    "evidence_provided": serializable_evidence_dict})
            # Save after each case
            try:
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)
                print(f"      Progress saved after case {actual_case_index_in_original_df + 1}. Total {len(all_results)} items in '{result_file_path}'.")
            except Exception as e:
                print(f"      [ERROR SAVING] Could not save progress after case {actual_case_index_in_original_df + 1}: {e}")
    if start_case_index < num_total_to_process:
        print(f"Exact inference process completed. Total results ({len(all_results)} cases) saved to: {result_file_path}")
    return all_results


def bayesian_inference_approximate(model, evidences_df, variable_name, n_samples=1000, seed=None):
    """
    Perform approximate inference using Gibbs Sampling for multiple cases.
    """
    print("\nPerforming approximate inference with Gibbs Sampling for multiple cases...")
    gibbs_sampler = GibbsSampling(model)
    all_results = []
    total_max_predictions = 1000
    batch_size = 7
    num_total_to_process = min(total_max_predictions, evidences_df.shape[0])
    print(f"Performing approximate inference for {num_total_to_process} cases, in batches of {batch_size} (samples per case: {n_samples})...")
    for batch_start_idx in range(0, num_total_to_process, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, num_total_to_process)
        current_batch_df = evidences_df.iloc[batch_start_idx:batch_end_idx]
        if current_batch_df.empty:
            continue
        print(f"  Processing batch (approximate): cases {batch_start_idx + 1} to {batch_end_idx} (of {num_total_to_process})...")
        for i in range(current_batch_df.shape[0]):
            actual_case_index_in_original_df = batch_start_idx + i
            evidence_dict = current_batch_df.iloc[i].to_dict()
            keys_to_delete = []
            for k, v in evidence_dict.items():
                if pd.isna(v):
                    keys_to_delete.append(k)
                    continue
                if isinstance(v, float) and v.is_integer():
                    evidence_dict[k] = int(v)
            for k_del in keys_to_delete:
                del evidence_dict[k_del]
            print(f"    Case {actual_case_index_in_original_df + 1}/{num_total_to_process} - Evidence (approx): {evidence_dict}")
            try:
                samples_df = gibbs_sampler.sample(evidence=evidence_dict, size=n_samples, seed=seed, show_progress=False)
                if variable_name in samples_df.columns:
                    predicted_value_series = samples_df[variable_name].mode()
                    if not predicted_value_series.empty:
                        result_value = predicted_value_series[0]
                        if hasattr(result_value, 'item'):
                            result_value = result_value.item()
                        all_results.append({variable_name: result_value})
                    else:
                        print(f"    [WARNING] Could not determine mode for '{variable_name}' in case {actual_case_index_in_original_df + 1}. Mode series is empty.")
                        all_results.append({
                            "error": f"No mode found for {variable_name} (empty mode series)",
                            "evidence_case_number": actual_case_index_in_original_df + 1,
                            "evidence_provided": evidence_dict
                        })
                else:
                    print(f"    [ERROR] Variable '{variable_name}' not found in generated samples for case {actual_case_index_in_original_df + 1}.")
                    all_results.append({
                        "error": f"Variable {variable_name} not in generated samples",
                        "evidence_case_number": actual_case_index_in_original_df + 1,
                        "evidence_provided": evidence_dict
                    })
            except Exception as e:
                print(f"    [ERROR] Error processing case {actual_case_index_in_original_df + 1} with evidence {evidence_dict} using Gibbs Sampling: {e}")
                all_results.append({"error": str(e), "evidence_case_number": actual_case_index_in_original_df + 1, "evidence_provided": evidence_dict})
    result_file_path = os.path.join('./results', 'inference_approx_gibbs_ypred_batch.json')
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"Approximate inference results ({len(all_results)} cases processed) saved to: {result_file_path}")
    except Exception as e:
        print(f"[ERROR] Could not save approximate inference results: {e}")
    return all_results


def metrics_to_dataframe(y_val, y_val_pred, type_inference, model, output_dir='./results'):
    """
    Calculate and save metrics (accuracy, precision, recall, F1) and confusion matrix for predictions.
    """
    val_metrics = {
        'model': model,
        'dataset': 'val',
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    }
    # Print metrics
    for metrics in [val_metrics]:
        print(f"\nMetrics ({metrics['dataset']}):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
    # Save metrics to CSV
    metrics_df = pd.DataFrame([val_metrics])
    metrics_file_path = os.path.join(output_dir, f'metrics_rb_classic_{model}_{type_inference}.csv')
    metrics_df.to_csv(metrics_file_path, index=False)
    print(f"\nMetrics saved to: {metrics_file_path}")
    # Confusion matrix
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Validation')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    conf_matrix_file_path = os.path.join('./plots', f'confusion_matrix_rb_classic_{model}_{type_inference}.png')
    plt.savefig(conf_matrix_file_path)
    plt.close()
    print(f"Confusion matrix saved to: {conf_matrix_file_path}")
    # Classification report
    class_report = classification_report(y_val, y_val_pred)
    report_path = os.path.join(output_dir, f'classification_report_rb_classic_{model}_{type_inference}.txt')
    with open(report_path, 'w') as f:
        f.write(class_report)
    print(f"Classification report saved to: {report_path}")


def main():
    """
    Main function to load data, model, perform inference, and save metrics.
    """
    # Set model name and path
    model = 'hc'  # Change to 'gemini' or other as needed
    model_path = './models/best_model_hill_climb_bic-d_330504_bDeuScore-5747335.12_edges_103_20250608_214125.pkl'  # Update as needed
    print(f"Loading model from: {model_path}")
    # Load datasets
    train_encoded = pd.read_csv('./datasets/train_encoded.csv')
    train_df = pd.read_csv('./datasets/train_df.csv')
    val_encoded = pd.read_csv('./datasets/val_encoded.csv')
    val_df = pd.read_csv('./datasets/val_df.csv')
    print("DataFrames loaded successfully.")
    # Remove NaNs
    print("\n[INFO] NaN count in training DataFrame:")
    print(train_encoded.isna().sum())
    train_encoded.dropna(inplace=True)
    val_encoded.dropna(inplace=True)
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model_rb = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at {model_path}")
        print("Please check the model file path and name.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    # Exclude nodes for gemini model if needed
    nodes_to_exclude = ["TRATAMIENTO_VICTIMA", "VIOLENCIA_ECONOMICA"]
    if model == "gemini":
        for node in nodes_to_exclude:
            if node in model_rb.nodes():
                print(f"Excluding node {node} from gemini model...")
                try:
                    model_rb.remove_cpds(model_rb.get_cpds(node))
                except:
                    pass
                model_rb.remove_node(node)
        val_encoded = val_encoded.drop(columns=nodes_to_exclude, errors='ignore')
    # Evaluate Bayesian Network model
    print("\nEvaluating Bayesian Network model...")
    target_variable = 'NIVEL_DE_RIESGO_VICTIMA'
    try:
        markov_blanket = model_rb.get_markov_blanket(target_variable)
        if not markov_blanket:
            print(f"[WARNING] Markov blanket for '{target_variable}' is empty. It may be disconnected or isolated.")
        print(f"Markov blanket of '{target_variable}': {markov_blanket}")
    except Exception as e:
        print(f"[ERROR] Could not get Markov blanket for '{target_variable}': {e}")
        print("Ensure the target variable exists in the model.")
        return
    missing_cols = [col for col in markov_blanket if col not in val_encoded.columns]
    if missing_cols:
        print(f"[ERROR] The following Markov blanket columns are missing in val_encoded: {missing_cols}")
        print("Cannot proceed with inference.")
        return
    if not markov_blanket:
        print(f"[WARNING] Markov blanket for '{target_variable}' is empty. Cannot select evidence columns.")
        evidences_to_predict = pd.DataFrame()
    else:
        evidences_to_predict = val_encoded[markov_blanket]
    if model == "gemini":
        evidences_to_predict = [
            {k: v for k, v in ev.to_dict().items() if k not in nodes_to_exclude}
            for _, ev in evidences_to_predict.iterrows()
        ]
    # Parameter learning
    model_rb = parameter_learning(model_rb, train_encoded)
    output_dir = './results'
    # Inference and metrics
    print("\nPreparing data for batch inference...")
    type_inference = ['Exact']  # Add 'Approximate' if needed
    for i in type_inference:
        print(f"\nSaving metrics for inference type: {i}")
        if i == 'Exact':
            if evidences_to_predict.empty and markov_blanket:
                print("[WARNING] No data in val_encoded for Markov blanket columns, or val_encoded is empty.")
            elif not evidences_to_predict.empty:
                print(f"Using {evidences_to_predict.shape[0]} rows from val_encoded for inference.")
                all_results = bayesian_inference_exact(model_rb, evidences_to_predict, target_variable, model)
                all_results = all_results[:100]
            else:
                print("No inference will be performed as no evidence data is prepared.")
                continue
            y_val_pred = [res[target_variable] if isinstance(res, dict) and target_variable in res else None for res in all_results]
            valid_idx = [i for i, val in enumerate(y_val_pred) if val is not None]
            y_val_pred = [y_val_pred[i] for i in valid_idx]
            y_val = val_encoded[target_variable].iloc[valid_idx]
            y_val = y_val.iloc[:100]
            print("\nSaving model metrics...")
            metrics_to_dataframe(y_val, y_val_pred, i, model, output_dir)
        elif i == 'Approximate':
            if evidences_to_predict.empty and markov_blanket:
                print("[WARNING] No data in val_encoded for Markov blanket columns, or val_encoded is empty.")
            elif not evidences_to_predict.empty:
                print(f"Using {evidences_to_predict.shape[0]} rows from val_encoded for inference.")
                all_results = bayesian_inference_approximate(model_rb, evidences_to_predict, target_variable)
            else:
                print("No inference will be performed as no evidence data is prepared.")
            y_val_pred = [res[target_variable] if isinstance(res, dict) and target_variable in res else None for res in all_results]
            valid_idx = [i for i, val in enumerate(y_val_pred) if val is not None]
            y_val_pred = [y_val_pred[i] for i in valid_idx]
            y_val = val_encoded[target_variable].iloc[valid_idx]
            print("\nSaving model metrics...")
            metrics_to_dataframe(y_val, y_val_pred, i, model, output_dir)

if __name__ == "__main__":
    main()
