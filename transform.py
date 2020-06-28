from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer, PowerTransformer, OneHotEncoder
from collections import Counter
import numpy as np
import streamlit as st 

transformer = dict({'standard': StandardScaler(), 
'normalizer': Normalizer(),
'quantile':QuantileTransformer(output_distribution='normal'),
'B_C':PowerTransformer(method='box-cox'), 
'Y_J':PowerTransformer()})

def tform(df, key):
    return transformer[key].fit_transform(df)

def topk_encoder(X, k=10):
    counts = Counter(X)
    most_common = dict(counts.most_common(k))
    topk = list(most_common.keys())
    return OneHotEncoder(categories=[topk], handle_unknown='ignore')

