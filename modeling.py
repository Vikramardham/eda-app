import streamlit as st
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import LocallyLinearEmbedding
from transform import * 

class Model:
    def __init__(self, model_select, prob_type):

        if 'MLP' in model_select:
            
            nlayer = st.sidebar.slider('# of Layers', 1, 10, 4)
            alpha = st.sidebar.slider('L2 Penalty', 0.0, 0.01, 0.0001)
            node_i = []
            
            activation = st.sidebar.selectbox('activation', ['identity', 'logistic', 'tanh', 'relu'], index=3)
            
            for i in range(1, nlayer):
                node_i.append(st.sidebar.slider(f'nodes in layer {i}', 1, 100, 10))
            
            if not prob_type:
            
                self.model = MLPClassifier(alpha=alpha, max_iter=100, hidden_layer_sizes=node_i, 
                activation=activation, learning_rate='adaptive', early_stopping=True)
            else:
            
                self.model = MLPRegressor(alpha=alpha, max_iter=100, hidden_layer_sizes=node_i,
                activation=activation, learning_rate='adaptive', early_stopping=True)

        if 'RandomForest' in model_select:
            
            n_estimators = st.sidebar.slider('# of estimators', 100, 500, 100)
            max_features = st.sidebar.selectbox('Max_features', ['auto', 'log2', 'sqrt'], index=0)
            max_depth = st.sidebar.slider('max_depth',1, 100, None)
            class_weight = st.sidebar.selectbox('Criterion', ['balanced', 'balanced_subsample'], index=1)
            
            if not prob_type:
            
                crit = st.sidebar.selectbox('Criterion', ['gini', 'entropy'], index=0)
                self.model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, 
                class_weight=class_weight, max_features=max_features)
            else:
            
                crit = st.sidebar.selectbox('Criterion', ['mse', 'mae'], index=0)
                self.model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, 
                oob_score=True, max_features=max_features)

        if 'SVM' in model_select:
            
            lam=st.sidebar.slider('C', 0.0, 1.0, 0.01)
            kernel = st.sidebar.selectbox('Kernel', ['linear', 'poly', 'rbf', 'sigmoid'], index=0)
            
            if not prob_type:
                self. model=SVC(C=lam, kernel=kernel, probability=True, class_weight='balanced')
            else:
                self.model=SVR(C=lam, kernel=kernel)
    
    def pipe(self, df, transforms, cat_cols, num_cols, dep_cols):

        steps = [('scale', transformer[transforms],
                        [num for num in num_cols if num in dep_cols])]
        
        for cat in cat_cols:
            if cat in dep_cols:
                steps.append((cat, topk_encoder(df[cat].values.tolist(), 10), [dep_cols.index(cat)]))

        pre_process = ColumnTransformer(steps)

        clf = Pipeline(steps=[('preprocessor', pre_process), 
                                ('classify', self.model)])
        return clf

    def get_performance(self):
        return
