from data_analysis import data_analysis
import pandas as pd
import numpy as np

def data_preprocess():
    df = data_analysis()
    features_na = [features for features in df.columns if df[features].isnull().sum() > 0]
    for feature in features_na:
        print(feature, np.round(df[feature].isnull().mean()))
    else:
        print("no missing value found")
    # Find Features with One Value
    for column in df.columns:
        print(column,df[column].nunique())
    #Exploring the Categorical Features
    categorical_features = [feature for feature in df.columns if ((df[feature].dtypes=='O') & (feature not in ['y']))]
    print(categorical_features)
    for feature in categorical_features:
        print('The feature is {} and number of categories are {}'.format(feature,len(df[feature].unique())))
    #list of numerical features
    numerical_features = [feature for feature in df.columns if ((df[feature].dtypes != 'O') & (feature not in ['y']))]
    print('Number of numerical variables:', len(numerical_features))
    #visualize the numerical variables
    #df[numerical_features].head()
    
    return df
    
data_preprocess()    
