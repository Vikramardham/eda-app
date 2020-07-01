import streamlit as st
import plotly.graph_objects as go
import seaborn as sns 
from utils import print_decorator, summarize_cat, box_plot, plotly_hist
from transform import *
from plotly.subplots import make_subplots
import math 
import ppscore as pps 

@st.cache(suppress_st_warning=True)
def class_dist(df, dep_var, prob_type):
    if not prob_type:
        # For classification problems
        print_decorator('Class Distribution')
        if not prob_type:
            classes = df[dep_var].unique()
            balance = df[dep_var].value_counts()
        
        fig = go.Figure(go.Bar(x=df[dep_var].value_counts().keys(), y=df[dep_var].value_counts().values))
        fig.update_layout(xaxis_title="Classes", yaxis_title="Counts",)
        st.plotly_chart(fig)
    else:
        # For regresssion problems
        sns.distplot(df[dep_var])
        st.pyplot()

@st.cache()
def cat_dist(df, cat_cols):
    if len(cat_cols)>0:
        cat_bar = make_subplots(rows=math.ceil(len(cat_cols)/3), cols=min(len(cat_cols),3), subplot_titles=cat_cols)
        for i, cat in enumerate(cat_cols):
            unique, popular, counts = summarize_cat(df[cat])
            if len(popular) > 1:
                cat_bar.add_trace(go.Bar(x=popular.index, y=popular.values), row=i//3+1, col=i%3+1)
            else:
                cat_bar.add_trace(go.Bar(x=['TOO MANY CATS'], y=[1.0]), row=i//3+1, col=i%3+1)
        cat_bar.update_layout(showlegend=False)

        return cat_bar

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def num_dist(df, num_cols, transforms):

    if transforms == 'Custom':
        custom_func = st.text_input('f(x)', 'x')
        print('Coming soon ...')
        # exec(f'func = lambda x: {custom_func}')
        # df.loc[:, num_cols] = tform(df[num_cols], transforms, func=func)
        # st.write(df.loc[:, num_cols])
    else:
        df.loc[:, num_cols] = tform(df[num_cols], transforms, func=None)

    
    #### Using boxplot, violin plot, density plot and cumulative plot
    log=False
    if st.checkbox('Log-scale'):
        log=True
    fig1 = box_plot(df, num_cols)
    fig2 = plotly_hist(df, num_cols, log=log)
    fig1.update_layout(showlegend=False)
    fig2.update_layout(showlegend=False)

    return fig1, fig2

@st.cache()
def pps_plot(df, columns, num_cols):
    pps_cache = st.cache(pps.matrix)
    corr = pps_cache(df)
    corr_fig = make_subplots(rows=1, cols=2, subplot_titles=('PPS','Standard Correlation'))
    corr_fig.add_trace(go.Heatmap(z=corr, x=columns, y=columns,
    coloraxis='coloraxis'), row=1, col=1)
    corr_fig.add_trace(go.Heatmap(z=df[num_cols].corr(), x=num_cols,y=None,
    coloraxis='coloraxis'), row=1, col=2)
    corr_fig.update_layout(coloraxis={'colorscale':'jet'})
    return corr_fig