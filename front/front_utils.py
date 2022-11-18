import streamlit as st
import numpy as np
import pandas as pd

def select_box_in_col(col, label, options):
    with col:
        val = st.selectbox(label, tuple(options))
    return val


def slider_in_col(col, label, min_, max_):
    with col:
        val = st.slider(label, min_value=min_, max_value=max_, value=max_)
    return val


def slider_in_col_v2(col, label, min_, max_):
    with col:
        val = st.slider(label, min_value=min_, max_value=max_, value=[min_, max_], step=1)
    return val


def markdown_title(title):
    markdown_text = f"<p style='text-align: center; color: black;'>{title}<br>.</p>"
    return st.markdown(markdown_text, unsafe_allow_html=True)

def markdown_title_with_expander(expander_title, title, expanded=True):    
    with st.expander(expander_title, expanded=expanded):
        return markdown_title(title)