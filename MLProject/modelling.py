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
    
    # Initialize DagsHub tracking
    # This switches our tracking URI to DagsHub
    dagshub.init(repo_owner='anwarrohmadi2006', repo_name='Eksperimen_SML_Anwar-Rohmadi', mlflow=True)
    
    # Set the experiment name explicitly
    mlflow.set_experiment('Eksperimen_SML_Anwar-Rohmadi')
    
    # Enable autologging
    mlflow.autolog(log_models=True)
    
    # We use explicit start_run() because mlflow project run was local-only.
    # This will create a fresh run on DagsHub.
    with mlflow.start_run():
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
        
        run_id = mlflow.active_run().info.run_id
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(MODEL_OUTPUT_DIR, "run_id.txt"), "w") as f:
            f.write(run_id)
        
        print(f"Training complete (DagsHub). RMSE: {rmse:.4f}")
        print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
