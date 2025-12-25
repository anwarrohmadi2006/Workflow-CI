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
    
    # ==== BULLETPROOF DAGSHUB INTEGRATION ====
    # Step 1: End any active run from 'mlflow run' context
    if mlflow.active_run():
        mlflow.end_run()
    
    # Step 2: Clear ALL MLflow environment variables that could cause conflicts
    for env_var in ['MLFLOW_RUN_ID', 'MLFLOW_EXPERIMENT_ID', 'MLFLOW_TRACKING_URI']:
        if env_var in os.environ:
            del os.environ[env_var]
    
    # Step 3: Set DAGSHUB_USER_TOKEN for dagshub.init() to use
    if os.getenv('DAGSHUB_TOKEN'):
        os.environ['DAGSHUB_USER_TOKEN'] = os.getenv('DAGSHUB_TOKEN')
        print("CI detected: Using DAGSHUB_TOKEN for authentication")
    
    # Step 4: Use dagshub.init() - this handles EVERYTHING correctly
    dagshub.init(repo_owner='anwarrohmadi2006', repo_name='Membangun_model', mlflow=True)
    print(f"DagsHub initialized: {mlflow.get_tracking_uri()}")
    
    # Step 5: Enable autologging BEFORE starting run
    mlflow.autolog(log_models=True)
    
    # Step 6: Start a fresh run
    with mlflow.start_run() as run:
        remote_run_id = run.info.run_id
        print(f"Started run: {remote_run_id}")
        
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
        
        # Save run_id for next CI steps
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(MODEL_OUTPUT_DIR, "run_id.txt"), "w") as f:
            f.write(remote_run_id)
        
        print(f"Training complete (DagsHub). RMSE: {rmse:.4f}")
        print(f"Run ID: {remote_run_id}")

if __name__ == "__main__":
    main()
