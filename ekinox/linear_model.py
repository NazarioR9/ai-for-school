import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from ekinox.model_utils import seed_everything

def train_linear_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    
    return model
    
def compute_raw_score(df, columns, coefs):
    y = 0
    for col, coef in zip(columns, coefs):
        y += coef * df[col] / df[col].max()
    
    return y

def compute_improvability_score(df, columns, model):
    # reduce improvability score for top tier students (student with  high grades)
    # and boost improvability for student in the middle and lowest 2/4 tiers
    grade_factor = 1 - ( (df['FinalGrade'] - df['FinalGrade'].min()) / df['FinalGrade'].max() )
    
    coefs = model.coef_ / model.coef_.sum()
    score = compute_raw_score(df, columns, coefs)
    score = score - score.min() / score.max()
    score = score * grade_factor
    
    return score

def end_to_end_linear_model(df):
    seed = 0
    seed_everything(seed)
    
    columns = ['studytime', 'traveltime', 'failures', 'freetime', 'goout', 'Walc', 'Dalc', 'health', 'absences']
    model = train_linear_model(df[columns], df['FinalGrade'])
    
    improvability_score = compute_improvability_score(df, columns, model)
    
    df['improvability'] = improvability_score
    
    return df