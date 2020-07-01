from imports import *

st.markdown('''
# <div class="alert alert-success" role="alert"> Exploratory Data Analysis</div>
''', unsafe_allow_html=True)

#### Upload a CSV datafile ####
data_file = st.file_uploader('Upload CSV')   
# if nothing is upload, a default file is used (a credit fraud dataset from Kaggle) ####
if not data_file:
    data_file = 'data/log.csv' 

#### Choose number of rows to read in the datafile ####
nrows = st.sidebar.slider('# of Samples to Read', min_value=1000, max_value=20000, value=1000) 
df_raw = read_dat(data_file, nrows=nrows) 
df = df_raw.copy(deep=True) # make a copy for better caching
df.columns = [re.sub(r'"', '', col) for col in df.columns] # remove quotes in column names (if they exist)
columns = df.columns.values.tolist()

#### Choose dependent variable, default = <last column> and the type of Problem  #### 
dep_var = st.sidebar.selectbox('Dependent Variable', columns, index=len(columns)-1)
prob_type = st.sidebar.checkbox('Regression Problem?')

#### Check the distribution of Output (to see how imbalanced the data is)
class_dist(df, dep_var, prob_type)
##### D-Tale app provides nice features to look at the data, check it out!
dtale_app.JUPYTER_SERVER_PROXY = True

if st.checkbox('Show DTale'):
    st.markdown('''### Does not work on Heroku (yet!)''')
    d = dtale.show(df, ignore_duplicate=True)
    st.write(f'<a target=\"_blank\" href=\"{d._main_url}\">Dtale</a>',  unsafe_allow_html=True)

#### 
st.markdown(''' > ## Data Types of Columns''')
st.write(pd.DataFrame(df.dtypes, columns=['Data Type']).T)
col_types, num_cols, cat_cols = get_dtypes(df)

### Take a look at missing data
st.markdown('> ## Missing Data')
msno.matrix(df)
st.pyplot()

#### Just drop missing data 
#### May need a better strategy for Modeling
df.dropna(inplace=True)

st.markdown('''> ## Checkout Data Distribution
>> ### Categorical Data''')
if st.checkbox('Choose Custom Categorical Columns'):
    cat_cols_new = st.multiselect('Choose Categorical Columns', cat_cols, default=cat_cols)
    num_cols = num_cols + [cat for cat in cat_cols if cat not in cat_cols_new]
    cat_cols = cat_cols_new
cat_bar = cat_dist(df, cat_cols)


st.plotly_chart(cat_bar)

#### Transform and View Numerical Data
transforms = st.selectbox('Apply Transformation', list(transformer.keys()))
fig1, fig2 = num_dist(df, num_cols, transforms)
st.markdown('''> ### Numerical Data ''')
st.plotly_chart(fig1)
st.plotly_chart(fig2)

#### Compare a couple of classes and their data distributions
if not prob_type:
    COMPARE_CLASSES=st.checkbox('Compare Classes')
    if COMPARE_CLASSES:
        st.markdown('''> ## Compare distributions of two target classes''')
        classes = df[dep_var].unique()
        cls1 = st.selectbox('Class ID 1', classes, index=0)
        cls2 = st.selectbox('Class ID 2', classes, index=1)
        
        fig1 = box_plot(df[df[dep_var]==cls1], num_cols)
        fig1.update_layout(title='Class #1')
        fig2 = box_plot(df[df[dep_var]==cls2], num_cols)
        fig2.update_layout(title='Class #2')
        
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

#### Check out the data topology using the fancy Mapper algorithm

SHOW_TOPOLOGY=st.checkbox('Show Data Topology Graph')
if SHOW_TOPOLOGY:
    st.markdown('''> ## Data Topology
    >> ### This represents our high-dimensional data on a 2D plane as a network''')
    
    if not prob_type:
        map = plot_mapper(df[num_cols], df[dep_var].astype('category').cat.codes)
    else:
        map = plot_mapper(df[num_cols+[dep_var]], color=None)
    st.plotly_chart(map)

#### Take a look at PPS and Standard correlation between various columns
#### Standard correlation is only defined for numerical data
st.markdown('''> ## Check correlations between the columns 
>> ### Let us go beyond standard correlation and view the Predicability Power Score''')
corr_fig = pps_plot(df, columns, num_cols)
st.plotly_chart(corr_fig)

#### Perform Basic Modeling for better Understanding of the data

BUILD_MODEL=st.checkbox('Build Models?')

if BUILD_MODEL:

    st.markdown('''> ## Let us build simple models''')
    dep_cols = st.sidebar.multiselect('Independent Variables', columns, columns[:-1])
    classifiers = [ 'SVM', 'RandomForest', 'MLP']
    model_select = st.sidebar.selectbox('Models', classifiers, index=0)

    modeling=Model(model_select, prob_type)

    X = df[dep_cols]
    y = df[dep_var]

    if not prob_type:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    clf = modeling.pipe(df, transforms, cat_cols, num_cols, dep_cols).fit(X, y)

    st.write('Score on the Test Data', clf.score(X_test, y_test))

# STOP HERE

st.balloons()