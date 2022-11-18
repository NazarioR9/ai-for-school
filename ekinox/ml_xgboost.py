import os
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import xgboost as xgb
from ekinox.model_utils import (
    seed_everything, select_categorical_columns, label_encode_features, compute_quantile, 
    kmeans_fit_predict, build_leaks_free_target_encoding
)
    
def load_model(path):
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model(path)

    return loaded_model

def make_splits(df, stratify_col, split=0.2, seed=0):
    train, val = train_test_split(df, test_size=split, stratify=df[stratify_col], random_state=seed)
    
    return train, val

def train_and_save_model(x_train, y_train, x_val, y_val, seed=0, path="xgboost_model_sklearn.json"):
    params = {
        'objective': 'reg:absoluteerror',
        'max_depth': 6,
        'n_estimators': 10000,
        'learning_rate': 0.01,
        'random_state': seed,
        'seed': seed,
        'early_stopping_rounds': 200,
        'verbose': 100,
        'eval_metric': 'mae'
    }

    model = xgb.XGBRegressor(**params)
    
    model.fit(
        x_train, y_train,
        eval_set = [(x_train, y_train), (x_val, y_val)],
        verbose=100
    )
    
    model.save_model(path)
    
    return model
    
def end_to_end_xgboost(df, retrain=False):
    seed = 0
    seed_everything(seed)
    
    categorical_columns = select_categorical_columns(df, ['FirstName', 'FamilyName'])
    df = label_encode_features(df, categorical_columns, copy=False)
    
    drop_columns = ['StudentID', 'FinalGrade', 'FirstName', 'FamilyName']
    X = df.drop(columns=drop_columns)
    
    df['cluster'] = kmeans_fit_predict(X, k=5)
    
    train, val = make_splits(df, 'cluster', seed=seed)
    
    te_data = build_leaks_free_target_encoding(train, 'cluster', 'FinalGrade')
    train = train.merge(te_data, on='cluster', how='left')
    val = val.merge(te_data, on='cluster', how='left')
    df = df.merge(te_data, on='cluster', how='left')
    
    x_train, y_train = train.drop(columns=drop_columns), train['FinalGrade']
    x_val, y_val = val.drop(columns=drop_columns), val['FinalGrade']
    
    model_path = "models/xgboost_model_sklearn.json"
    model = None
    
    if os.path.exists(model_path) and not retrain:
        model = load_model(model_path)
    
    if model is None or retrain:
        model = train_and_save_model(x_train, y_train, x_val, y_val, seed, model_path)
    
    df['Predicted'] = model.predict(df.drop(columns=drop_columns))
    df['Diff'] = df['Predicted'].apply(round) - df['FinalGrade']
    
    return df
