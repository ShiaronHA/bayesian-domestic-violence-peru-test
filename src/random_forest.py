from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from metrics import save_metrics

# --- Paths ---
TRAIN_ENCODED_PATH = './datasets/train_encoded.csv'
VAL_ENCODED_PATH   = './datasets/val_encoded.csv'
OUTPUT_DIR         = './results'

# --- Model configuration ---
TARGET_VARIABLE          = 'NIVEL_DE_RIESGO_VICTIMA'
RANDOM_STATE             = 42
FEATURE_IMPORTANCE_THRESHOLD = 0.01  # Minimum feature importance to retain a feature


def feature_selection(train, X_val, target_col):
    X = train.drop(target_col, axis=1)
    y = train[target_col]

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf.fit(X, y)

    importances = rf.feature_importances_
    selected_features = X.columns[importances > FEATURE_IMPORTANCE_THRESHOLD]

    print("Features selected by RandomForest:")
    print(selected_features)

    X_val_aligned = X_val.reindex(columns=X.columns, fill_value=0)
    X_train_selected = X[selected_features]
    X_val_selected = X_val_aligned[selected_features]

    return X_train_selected, X_val_selected


def learn_with_random_forest(train, target_col, val, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val = val.drop(columns=[target_col])
    y_val = val[target_col]

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")

    X_train, X_val = feature_selection(train, X_val, target_col)

    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)

    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)

    save_metrics(y_val, y_val_pred, 'random_forest', 'RandomForest', output_dir)

    return rf_model


def load_datasets():
    """Loads and cleans the train and validation encoded DataFrames."""
    train = pd.read_csv(TRAIN_ENCODED_PATH)
    val   = pd.read_csv(VAL_ENCODED_PATH)
    train.dropna(inplace=True)
    val.dropna(inplace=True)
    return train, val


def main():
    """Orchestrates the Random Forest training pipeline."""
    train, val = load_datasets()
    learn_with_random_forest(train, TARGET_VARIABLE, val, OUTPUT_DIR)


if __name__ == "__main__":
    main()
