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
    
    with mlflow.start_run():
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        
        model = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_OUTPUT_DIR, "model")
        mlflow.sklearn.save_model(model, model_path)
        mlflow.sklearn.log_model(model, "model", registered_model_name="house_prices_model")
        
        run_id = mlflow.active_run().info.run_id
        with open(os.path.join(MODEL_OUTPUT_DIR, "run_id.txt"), "w") as f:
            f.write(run_id)
        
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        print(f"Model saved to {model_path}")
        print(f"Run ID: {run_id}")

if __name__ == "__main__":
    main()
