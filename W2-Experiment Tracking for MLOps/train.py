import os
import pickle
import argparse
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_training(data_path):
    with mlflow.start_run():
        mlflow.sklearn.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
        X_val, y_val = load_pickle(os.path.join(data_path, 'val.pkl'))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"Validation RMSE: {rmse}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the processed data folder')
    args = parser.parse_args()
    run_training(args.data_path)