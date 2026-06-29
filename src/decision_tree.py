from sklearn.tree import DecisionTreeClassifier
import os
import pandas as pd
from metrics import save_metrics

# --- Paths ---
_BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ENCODED_PATH = os.path.join(_BASE_DIR, 'datasets', 'train_encoded.csv')
VAL_ENCODED_PATH   = os.path.join(_BASE_DIR, 'datasets', 'val_encoded.csv')
OUTPUT_DIR         = os.path.join(_BASE_DIR, 'results')

# --- Model configuration ---
TARGET_VARIABLE              = 'NIVEL_DE_RIESGO_VICTIMA'
RANDOM_STATE                 = 42
FEATURE_IMPORTANCE_THRESHOLD = 0.01
MAX_PREDICTIONS              = 100       # Must match bayesian_inference.py MAX_PREDICTIONS for a fair benchmark


def feature_selection(train, X_val, target_col):
    X = train.drop(target_col, axis=1)
    y = train[target_col]

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X, y)

    importances = dt.feature_importances_
    selected_features = X.columns[importances > FEATURE_IMPORTANCE_THRESHOLD]

    print("Features selected by DecisionTree:")
    print(selected_features)

    X_val_aligned = X_val.reindex(columns=X.columns, fill_value=0)
    X_train_selected = X[selected_features]
    X_val_selected = X_val_aligned[selected_features]

    return X_train_selected, X_val_selected


def learn_with_decision_tree(train, target_col, val, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col]).iloc[:MAX_PREDICTIONS]
    y_val = val[target_col].iloc[:MAX_PREDICTIONS]

    print(f"X_val limited to first {MAX_PREDICTIONS} cases for benchmark parity.")
    X_train, X_val = feature_selection(train, X_val, target_col)

    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)

    y_train_pred = dt_model.predict(X_train)
    y_val_pred = dt_model.predict(X_val)

    save_metrics(y_val, y_val_pred, 'decision_tree', 'DecisionTree', output_dir)

    return dt_model


def load_datasets():
    """Loads and cleans the train and validation encoded DataFrames."""
    train = pd.read_csv(TRAIN_ENCODED_PATH)
    val   = pd.read_csv(VAL_ENCODED_PATH)
    train.dropna(inplace=True)
    val.dropna(inplace=True)
    return train, val


def main():
    """Orchestrates the Decision Tree training pipeline."""
    train, val = load_datasets()
    learn_with_decision_tree(train, TARGET_VARIABLE, val, OUTPUT_DIR)


if __name__ == "__main__":
    main()
