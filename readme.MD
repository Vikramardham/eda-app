# A Simple Data Exploratory App using Streamlit

1. Provides interactive tools to explore data, normalize, scale and view outliers

2. Provides simple modeling strategies to understand the data better

3. It should be used ONLY to obtain a better understanding of the underlying data and not to develop models

4. Reasonable only with dataset with few columns (~20)

5. The script is deployed as an app @ https://arcane-headland-76563.herokuapp.com/

> ## Usage Description

+ Install streamlit, pps_score, giotto-tda, D-Tale, Missingno and other dependencies
+ Clone the repository: ```git clone https://github.com/Vikramardham/eda-app.git```
+ Run as ```streamlit run eda.py```
+ Upload your favorite csv file
+ Check out  

> ## Features

1. File upload
    + Upload your favorite csv file with categorical and numerical columns
    + Choose your dependent variable and the rest are considered independent
    + Numerical and Categorical data types are guessed based on data type and frequency
2. Shows Missing values using the *missingno* library (5)
3. Displays distributions of categorical and numerical data
4. Numerical data can be transformed and viewed simultaneously
    + The popular transformations in scikit-learn are implemented
    + It is trivial to add an additional transformer (in *transform.py*)
5. Choose and compare data distributions (only numerical) of two target classes
6. The numerical data can be projected on two a 2D plane as a graph using the *Mapper algorithm* (2)
    + This projection allows visualization of complex data topology
    + Can help figure out outliers and anamolies
    + Coloring appropriately provides class separation
7. View correlations among the data and Predictive Power Score (1) Matrix
8. Develop a simple model to understand your data better
    + *Random Forest*, *SVM* and *MLP* are available for now
    + Additional models can be added rather trivially in *modeling.py*
9. Allows tuning of your model and outputs a simple performance score on the test dataset

> ## Intented Usage

1. I often find exploring a new dataset quite cumbersome and repetititve, so I decided to write a simple app to save time and make the process more enjoyable.
2. Also, I wanted to try out Streamlit (3). Streamlit is quite amazing and there is a lot more features we can look forward to!
3. This is built purely for personal use and might not work for everyone but I intend to add more features as I go!
4. I took some inspiration from pandas-profile and sweetviz and added what I thought was missing!

> ## Future Plans

1. Implement Feature importance (SHAP/LIME) 
2. Compute Feature Interactions of various column pairs towards predictability 
3. Ensemble Models

> ## Snapshots

![Snap1](images/top.PNG)
- - - - 
![Snap2](images/top_regression.PNG)
- - - - 
![Snap3](images/mid_regression.PNG)

References:

1. [Predictive Power Score](https://github.com/8080labs/ppscore)
2. [Mapper Algorithm](https://arxiv.org/abs/2004.02551)
3. [Streamlit](https://github.com/streamlit/streamlit)
4. [D-Tale](https://github.com/man-group/dtale)
5. [Missingno](https://github.com/ResidentMario/missingno)
