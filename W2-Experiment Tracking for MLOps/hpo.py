import os
import pickle
import argparse
import mlflow
from hyperopt import fmin, tpe, hp, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def objective(params, X_train, y_train, X_val, y_val):
    with mlflow.start_run():
        mlflow.log_params(params)
        rf = RandomForestRegressor(**params, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
    return rmse

def run_hpo(data_path):
    X_train, y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
    X_val, y_val = load_pickle(os.path.join(data_path, 'val.pkl'))

    search_space = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'n_estimators': hp.choice('n_estimators', range(10, 150)),
        'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
    }

    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, X_train, y_train, X_val, y_val),
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
    )
    print(f"Best parameters: {best}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the processed data folder')
    args = parser.parse_args()
    run_hpo(args.data_path)