import numpy as np 
import pandas as pd 
import dtale
import streamlit as st 
import altair as alt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import missingno as msno
import sys 
import os
import time
import math
from utils import *
from transform import *
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline

import ppscore as pps
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

data_file = 'data/log.csv' # add options for uploading
# %% add file creation time
date = time.ctime(os.path.getctime('eda.py'))

st.markdown('''
> # Exploratory Data Analysis
''')
st.write(date)

#show_image('bg.jpg', caption='Background')
#%% side bar
nrows = st.sidebar.slider('# of rows', min_value=100, max_value=10000, value=1000) 
## chose proper min and max row values automatically
df_raw = read_dat(data_file, nrows=nrows)
df = df_raw.copy(deep=True)
columns = df.columns.values.tolist()

dep_var = st.sidebar.selectbox('Dependent Variable', columns, index=len(columns)-1)
prob_type = st.sidebar.checkbox('Regression Problem?')

print_decorator('Class Distribution')
if not prob_type:
    classes = df[dep_var].unique()
    balance = df[dep_var].value_counts()
sns.countplot(df[dep_var])
st.pyplot()

# %%
st.write(':sunglasses:')
if st.checkbox('Show DTale'):
    d = dtale.show(df, ignore_duplicate=True)
    st.write(f'<a target=\"_blank\" href=\"//vrapc:40000/dtale/main/1\">Dtale</a>',  unsafe_allow_html=True)
st.write(':sunglasses:')
# %%
st.markdown(''' > ## Types of Columns''')
st.write(df.dtypes)
col_types, num_cols, cat_cols = get_dtypes(df)
#st.write('Numerical Columns')
#st.write(num_cols[0])
#st.write('categorical Columns')
#st.write(cat_cols[0])
#st.write(col_types['float'])
#st.write(col_types['object'])
#df_cols = pd.concat([pd.DataFrame(num_cols, index=['Numerical']), 
#pd.DataFrame(cat_cols, index=['Categorical'])])
#st.write(df_cols)
# %% altair plots for missing values and nans
st.markdown('> ## Missing Values')
msno.matrix(df)
st.pyplot()
#%%
# what do with the missing data?
#st.write(df.isna())
#imputer = IterativeImputer(random_state=0, max_iter=100)
#df.loc[:, num_cols[0]] = imputer.fit_transform(df[num_cols[0]])

# %% Fit to common distributions
st.markdown(''' ## Checkout Data Distribution''')
cat_bar = make_subplots(rows=math.ceil(len(cat_cols[0])/3), cols=3)
for i, cat in enumerate(cat_cols[0]):
    #st.dataframe((df[cat].value_counts()).transpose())
    unique, popular, counts = summarize_cat(df[cat])
    if len(popular) > 1:
        cat_bar.add_trace(go.Bar(x=popular.index, y=popular.values), row=i//3+1, col=i%3+1)
    else:
        cat_bar.add_trace(go.Bar(x=['TOO MANY CATS'], y=[1.0]), row=i//3+1, col=i%3+1)

st.plotly_chart(cat_bar)
#select = 
select_row = st.selectbox('Show Columns', [*num_cols[0], 'Show All'], index=0)
if select_row!='Show All':
    charts = plot_hist(df, select_row, classes, dep_var)
    st.altair_chart(alt.hconcat(*charts))
else:
    charts = []
    for cols in num_cols[0]:
        charts.extend(plot_hist(df, cols, classes, dep_var))
    out = facet_wrap(charts, 3)
    st.altair_chart(out)
    
transforms = st.selectbox('Transform Data', list(transformer.keys()))
df.loc[:, num_cols[0]] = tform(df[num_cols[0]], transforms)
df
### Using boxplot, violin plot, density plot and cumulative plot
log=False
if st.checkbox('Log-scale'):
    log=True
fig1 = box_plot(df, num_cols[0])
fig2 = plotly_hist(df, num_cols[0], log=log)
st.plotly_chart(fig1)
st.plotly_chart(fig2)

# compare classes
cls1 = st.selectbox('Class ID 1', classes, index=0)
cls2 = st.selectbox('Class ID 2', classes, index=1)
fig1 = box_plot(df[df[dep_var]==cls1], num_cols[0])
fig2 = box_plot(df[df[dep_var]==cls2], num_cols[0])
st.plotly_chart(fig1)
st.plotly_chart(fig2)

## mapper and t-sne for topology views and outlier detection
map = plot_mapper(df[num_cols[0]], df[dep_var].values)
st.plotly_chart(map)

#%% correlations 
# among columns
# with the dependent variable (LDA)
pps_cache = st.cache(pps.matrix)
corr = pps_cache(df)
corr
corr_fig = make_subplots(rows=1, cols=2)
corr_fig.add_trace(go.Heatmap(z=corr, x=columns, y=columns,
coloraxis='coloraxis'), row=1, col=1)
corr_fig.add_trace(go.Heatmap(z=df[num_cols[0]].corr(), x=num_cols[0],y=None,
 coloraxis='coloraxis'), row=1, col=2)
corr_fig.update_layout(coloraxis={'colorscale':'jet'})
st.plotly_chart(corr_fig)

#%% modeling
classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),]

model = st.sidebar.selectbox('Models', classifiers, index=0)

dep_cols = st.multiselect('Independent Variables', columns, num_cols[0])

steps = [('scale', transformer[transforms],
                        make_column_selector(dtype_include=np.number))
                        ]
for cat in cat_cols[0]:
    #st.write(df[cat].values.tolist())
    if cat in dep_cols:
        steps.append((cat, topk_encoder(df[cat].values.tolist(), 10), [dep_cols.index(cat)]))

pre_process = ColumnTransformer(steps)

clf = Pipeline(steps=[('preprocessor', pre_process), 
                        ('classify', model)])

X = df[dep_cols]
y = df[dep_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
clf.fit(X_train, y_train)

st.write(clf.score(X_test, y_test))


if 'MLPClassifier' in model.__repr__():
    nlayer = st.sidebar.slider('# of Layers', 1, 10)
    node_i = []
    for i in range(1, nlayer):
        node_i.append(st.sidebar.slider(f'nodes in layer {i}', 1, 100))
#  GLM, xgboost, MLP
# STOP HERE


# FUTURE 
# dimensionality reduction
# guess the distribution of data 
# interactions of variables toward output prediction 
# ensemble (hybrid) models 
# Naive-Bayes, SVM
