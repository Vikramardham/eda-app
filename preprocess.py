import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns 
import plotly.figure_factory as ff
st.title('Fraud Detection')
st.markdown('''> ## Experiments with Streamlit in building a Fraud Detection Algorithm on Synthetic Dataset from Kaggle
Source: https://www.kaggle.com/ntnu-testimon/paysim1 

Author: Vikram Reddy Ardham 
June 04, 2020
----
''')

# Build a EDA app
# take input and other arguments (how much of sample data)
# 1. view data
# 2. summarize categorical and numerical variables
# 2a. show missing values and nans
# 3. build distribution plots
# 4. build correlation plots (versus each other and dep. variable)
# 5. build a simple model
# 6. use other libs like Dtale 
# 7. provide options to fiddle the data ... checkbox, sliders, dropbox, animations
# 8. Simple Modeling Options

st.write('## EDA')
source = '../data/log.csv'

@st.cache
def read_csv(source):
    return pd.read_csv(source, sep='\t')

data_sample = read_csv(source)
st.write('Df', data_sample.head())

st.write(data_sample.describe())

# Figure
# missing values
sns.heatmap(data_sample.isna(), cbar=False)
st.pyplot()
st.write('missing values')
st.write(data_sample.isna().sum())
st.write(':sunglasses:')
#plt.plot(data_sample['amount'], data_sample['oldbalanceOrg'], 'o')
#st.pyplot()
fig = px.scatter(data_sample, x='amount', y='oldbalanceOrg', color='newbalanceOrig', size='oldbalanceDest')
st.plotly_chart(fig)

#correlation 
corr=data_sample.corrwith(data_sample['isFraud']).dropna()
#corr = corr.reset_index(level=0, inplace=True)
st.dataframe(corr)
fig1 = px.bar(x=corr.index, y = corr)
st.plotly_chart(fig1)
#st.dataframe(data_sample['isFlaggedFraud'].value_counts())

sns.heatmap(data_sample.corr())
st.pyplot()

# measure imbalance 
pos = (data_sample['isFraud']==1).sum()
neg = (data_sample['isFraud']==0).sum()
st.write('Postive Values:%d, Negative Values: %d' % (pos, neg))

# feature engineer (later)
# scale values 
st.dataframe(data_sample[data_sample['isFraud']==1])
pos_vals = (data_sample['isFraud']==1)
neg_vals = (data_sample['isFraud']==0)

num_cols = data_sample.columns[[2, 4, 5, 7, 8]].to_list()
st.write(num_cols)

#st.write(data_sample[num_cols].values.shape, num_cols)
#for col in num_cols:
#    st.write(len(data_sample[col].values.tolist()))

dists = data_sample[num_cols]
fig = ff.create_distplot([data_sample[c] for c in num_cols], num_cols)
st.plotly_chart(fig)
# modeling
# scale values
# metrics
# pipeline
# glm, mlp, randomforest, SVM