import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def register_best_model(mlflow_client, experiment_name, model_name):
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=5,
    )

    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(best_model_uri, model_name)
    print(f"Model registered: {model_name}")

if __name__ == '__main__':
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    EXPERIMENT_NAME = "random-forest-best-models"
    MODEL_NAME = "best-random-forest"

    register_best_model(client, EXPERIMENT_NAME, MODEL_NAME)register_model.py