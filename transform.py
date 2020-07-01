from sklearn.preprocessing import StandardScaler, Normalizer, QuantileTransformer, PowerTransformer, OneHotEncoder, FunctionTransformer
from collections import Counter
import numpy as np
import streamlit as st 

transformer = dict({'Standard Scaler': StandardScaler(), 
'Normalizer (Unit Norm)': Normalizer(),
'Quantile-Transformer':QuantileTransformer(output_distribution='normal'),
'Box-Cox':PowerTransformer(method='box-cox'), 
'Yeo-Johnson':PowerTransformer(),
'Custom': 'PlaceHolder'}) # add a new transformer of choice

def tform(df, key, func):
    if func:
        return FunctionTransformer(func).fit_transform(df)
    else:
        return transformer[key].fit_transform(df)

def topk_encoder(X, k=10):
    counts = Counter(X)
    most_common = dict(counts.most_common(k))
    topk = list(most_common.keys())
    return OneHotEncoder(categories=[topk], handle_unknown='ignore')

