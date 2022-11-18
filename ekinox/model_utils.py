import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def select_categorical_columns(df, exclude_columns=[]):
    return df.select_dtypes(exclude=np.number).columns.difference(exclude_columns).tolist()

def label_encode_features(df, columns, copy=False):
    data = df.copy() if copy else df
    for c in columns:
        le = LabelEncoder()
        data[c] = le.fit_transform(data[c])
    
    return data

def compute_quantile(q):
    def __quantile(data):
        return np.quantile(data, q)
    return __quantile

def scale_gain_within_cluster(df, cluster_column, column_to_scale):
    min_ = df.groupby(cluster_column)[column_to_scale].transform(np.min)
    max_ = df.groupby(cluster_column)[column_to_scale].transform(np.max)
    df[f'scale_{column_to_scale}'] = (df[column_to_scale] - min_) / max_

    return df

def groupby_transform(df, by_column, on_column, new_column_name, function):
    df[new_column_name] = df.groupby(by_column)[on_column].transform(function)
    
def build_leaks_free_target_encoding(df, by_column, on_column):
    te_data = df.groupby(by_column).agg(
        quantile_25 = (on_column, compute_quantile(0.25)),
        quantile_50 = (on_column, compute_quantile(0.5)),
        quantile_80 = (on_column, compute_quantile(0.8)),
    ).reset_index()
    
    return te_data

def kmeans_fit_predict(X, k):
    kmean = KMeans(n_clusters=k)
    kmean.fit(X)
    
    return kmean.predict(X)