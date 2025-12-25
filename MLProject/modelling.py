"""
MLflow Project Training Script for CI/CD
Author: Anwar-Rohmadi
"""
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub

DATA_DIR = "house_prices_preprocessing"
MODEL_OUTPUT_DIR = "model_output"

def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv").values.ravel()
    return X_train, X_val, y_train, y_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    
    X_train, X_val, y_train, y_val = load_data()
    
    # Initialize DagsHub tracking internally
    # This allows the script to log to DagsHub even if the MLflow Project context is local
    if os.getenv('DAGSHUB_TOKEN'):
        mlflow.set_tracking_uri("https://dagshub.com/anwarrohmadi2006/Eksperimen_SML_Anwar-Rohmadi.mlflow")
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME', 'anwarrohmadi2006')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
        print("CI detected: Scaling internal logging to DagsHub.")
    else:
        dagshub.init(repo_owner='anwarrohmadi2006', repo_name='Eksperimen_SML_Anwar-Rohmadi', mlflow=True)
    
    # Try to set experiment
    try:
        mlflow.set_experiment('Eksperimen_SML_Anwar-Rohmadi')
    except Exception as e:
        print(f"DagsHub Experiment Sync Note: {e}")
    
    # Enable autologging
    mlflow.autolog(log_models=True)
    
    # We use an explicit start_run to ensure the run is created ON DAGSHUB
    with mlflow.start_run() as run:
        remote_run_id = run.info.run_id
        
        model = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Log model with explicit registration
        mlflow.sklearn.log_model(
            model, 
            "model", 
            registered_model_name="house_prices_model"
        )
        
        # CRITICAL: We save the REMOTE run_id (DagsHub) for the next CI steps
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(MODEL_OUTPUT_DIR, "run_id.txt"), "w") as f:
            f.write(remote_run_id)
        
        print(f"Training complete (Remote DagsHub). RMSE: {rmse:.4f}")
        print(f"Remote Run ID: {remote_run_id}")

if __name__ == "__main__":
    main()
