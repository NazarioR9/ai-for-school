import pandas as pd
from ekinox.model_utils import (
    seed_everything, select_categorical_columns, label_encode_features, compute_quantile
)


def test_categorical_columns():
    df = pd.DataFrame({'col1': [0,1,2,3], 'col2': [0,1,2,3], 'col3': [0,1,2,3], 'col4': [0,1,2,3]})
    cat_cols = select_categorical_columns(df)
    
    assert len(cat_cols) == 0
    
    df = pd.DataFrame({'col1': [0,1,2,3], 'col2': [0,1,2,3], 'col3': [0,1,2,3], 'col4': ['a','b','c','d']})
    cat_cols = select_categorical_columns(df)
    
    assert len(cat_cols) == 1
    assert cat_cols == ['col4']

def test_label_encoder():
    df = pd.DataFrame({'col1': [0,1,2,3], 'col2': [0,1,2,3], 'col3': [0,1,2,3], 'col4': ['a','b','c','d']})
    df_le = label_encode_features(df, ['col4'], copy=True)
    assert (df_le['col4'] != df['col4']).mean() == 1
    
    df_le = label_encode_features(df, ['col4'], copy=False)
    assert df_le['col4'].tolist() == [0, 1, 2, 3]
    
def test_quantile_output():
    q_fct = compute_quantile(0.2)
    assert str(type(q_fct)) == "<class 'function'>"
    
