import pandas as pd
import streamlit as st 
import altair as alt 
import plotly.graph_objects as go
import plotly.figure_factory as ff
# tda magic
from gtda.mapper.filter import Projection
from gtda.mapper.cover import OneDimensionalCover, CubicalCover
from gtda.mapper.pipeline import make_mapper_pipeline
from gtda.mapper.visualization import plot_static_mapper_graph, plot_interactive_mapper_graph
from sklearn.cluster import DBSCAN
from PIL import Image

def show_image(image, caption=None, width=500):
    img = Image.open(image)
    st.image(img, caption=caption, use_column_width=False, width=width)
    return 
def status_decorator(func):
    def wrapper(*args, **kwargs):
        status = st.empty()
        status.text('Loading function %s' % func.__name__)
        out = func(*args, **kwargs)
        status.text('Done Loading function %s' % func.__name__)
        return out 
    return wrapper
def print_decorator(txt):
    #st.markdown('''----''')
    st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
    st.markdown(f'''> ## {txt}''')
    #st.markdown('''----''')
    return

@status_decorator
@st.cache
def read_dat(data_file, sep=None, nrows=100, **kwargs):
    if not sep:
        sep='\s|\t|,|;'
    else:
        sep=sep+'|'+'\s|\t|,|;'
    return pd.read_csv(data_file, sep=sep, nrows=nrows, engine='python', **kwargs)

def get_dtypes(df):
    '''return column data types of a dataframe '''
    cols = dict({})
    data_types = [ 'int64', 'float64', 'bool', 'object' ]
    for dtype in data_types:
        filter = df.select_dtypes(include=dtype).columns.values
        #st.write(filter)
        if len(filter)>0:
            cols.update({dtype: filter})
    
    num_cols = []
    cat_cols = []

    for key, val in cols.items():
        if key == 'float64':
            num_cols.append(val)
        if key == 'object':
            cat_cols.append(val)
    return cols, num_cols, cat_cols

@st.cache
def plot_hist(df, select_row, classes, dep_var):
    c1=[]
    c2=[]
    c3=[]
    
    c1 =alt.Chart(df, height=200, width=200).mark_bar(opacity=0.4, color='#EB6638').encode(
        alt.X(select_row, bin=True), y=alt.Y('count()', scale=alt.Scale(type='log'), axis=alt.Axis(title='Den'))).interactive()
    
    c2 = alt.Chart(df, height=200, width=200).transform_density(
        select_row, counts=True, as_=[select_row, 'density']).mark_area(opacity=0.4, color='#38EB89').encode(
            alt.X(select_row, axis=alt.Axis()), y=alt.Y('density:Q', axis=alt.Axis(title=None))).interactive()
    c3 = alt.Chart(df, height=200, width=200).transform_density(select_row, counts=True, cumulative=True, 
        as_=[select_row, 'density']).mark_area(opacity=0.4, color='#38E0EB').encode(
            alt.X(select_row, axis=alt.Axis()), y=alt.Y('density:Q', axis=alt.Axis(title=None))).interactive()
     
    return [c1, c2, c3]

# make a single row
@st.cache
def make_hcc(row_of_charts):
    hconcat = [chart for chart in row_of_charts]
    hcc = alt.HConcatChart(hconcat=hconcat, bounds='flush', spacing=40.)
    return hcc

# take an array of charts and produce a facet grid
@st.cache
def facet_wrap(charts, charts_per_row):
    rows_of_charts = [
        charts[i:i+charts_per_row] 
        for i in range(0, len(charts), charts_per_row)]        
    vconcat = [make_hcc(r) for r in rows_of_charts]    
    vcc = alt.VConcatChart(vconcat=vconcat, padding={'left':20, 'top':5}, spacing=40)\
      .configure_axisX(grid=True)\
      .configure_axisY(grid=True)
    return vcc
@st.cache
def box_plot(df, cols):
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Violin(y=df[col],
                            name=col,
                            box_visible=True,
                            meanline_visible=True))
    return fig
@st.cache
def plotly_hist(df, cols, log=False):
    fig = ff.create_distplot([df[c] for c in cols], cols , bin_size=.2)
    if log:
        fig.update_layout(xaxis_type="log", yaxis_type="log")
    return fig
@st.cache
def plot_mapper(df, color):
    pipeline = make_mapper_pipeline(
        filter_func=Projection(columns=2),
        cover=OneDimensionalCover(),
        clusterer=DBSCAN(),
    )
    return plot_static_mapper_graph(pipeline, df.values, color_variable=color)
@st.cache
def summarize_cat(series):
    counts = series.value_counts()
    unique = len(counts.keys())

    popular = pd.Series({key: val for key, val in counts.items() if val >2 })

    return unique, popular, counts