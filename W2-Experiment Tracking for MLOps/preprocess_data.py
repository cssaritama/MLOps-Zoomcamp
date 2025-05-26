import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle

def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

def preprocess_data(raw_data_path, dest_path):
    # Leer datos
    df_train = read_dataframe(os.path.join(raw_data_path, 'green_tripdata_2023-01.parquet'))
    df_val = read_dataframe(os.path.join(raw_data_path, 'green_tripdata_2023-02.parquet'))
    df_test = read_dataframe(os.path.join(raw_data_path, 'green_tripdata_2023-03.parquet'))

    # Definir características y objetivo
    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()
    train_dicts = df_train[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train['duration'].values

    val_dicts = df_val[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df_val['duration'].values

    test_dicts = df_test[categorical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_test = df_test['duration'].values

    # Guardar datos preprocesados
    os.makedirs(dest_path, exist_ok=True)
    with open(os.path.join(dest_path, 'dv.pkl'), 'wb') as f:
        pickle.dump(dv, f)

    with open(os.path.join(dest_path, 'train.pkl'), 'wb') as f:
        pickle.dump((X_train, y_train), f)

    with open(os.path.join(dest_path, 'val.pkl'), 'wb') as f:
        pickle.dump((X_val, y_val), f)

    with open(os.path.join(dest_path, 'test.pkl'), 'wb') as f:
        pickle.dump((X_test, y_test), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', type=str, help='Path to the raw data folder')
    parser.add_argument('--dest_path', type=str, help='Path to save processed data')
    args = parser.parse_args()
    preprocess_data(args.raw_data_path, args.dest_path)