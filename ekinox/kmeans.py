import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from ekinox.model_utils import (
    seed_everything, select_categorical_columns, label_encode_features, compute_quantile, scale_gain_within_cluster,
    groupby_transform, kmeans_fit_predict
)
    

def end_to_end_kmeans(df, cluster_k=5):
    seed_everything(seed=0)
    
    categorical_columns = select_categorical_columns(df, ['FirstName', 'FamilyName'])
    df = label_encode_features(df, categorical_columns, copy=False)
    drop_columns = ['StudentID', 'FinalGrade', 'FirstName', 'FamilyName']
    
    X = df.drop(columns=drop_columns)
    df['cluster'] = kmeans_fit_predict(X, cluster_k)
    
    groupby_transform(df, 'cluster', 'FinalGrade', 'cluster_mean_grade', np.mean)
    groupby_transform(df, 'cluster', 'FinalGrade', 'cluster_max_grade', np.max)
    groupby_transform(df, 'cluster', 'FinalGrade', 'cluster_top_tier_grade', compute_quantile(0.8))
    
    df['immediate_gain'] = (df['cluster_mean_grade'] - df['FinalGrade']).apply(lambda x: max(0., x))
    df['potential_gain'] = (df['cluster_top_tier_grade'] - df['FinalGrade']).apply(lambda x: max(0., x))
    df['maximum_gain'] = df['cluster_max_grade'] - df['FinalGrade']
    
    
    scale_gain_within_cluster(df, 'cluster', 'immediate_gain')
    scale_gain_within_cluster(df, 'cluster', 'potential_gain')
    scale_gain_within_cluster(df, 'cluster', 'maximum_gain')
    
    df['final_gain_scale'] = (df['scale_immediate_gain'] + df['scale_potential_gain'] + df['scale_maximum_gain']) / 3