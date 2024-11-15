import numpy as np
import pandas as pd
from datavisualization import visualise_data

def feature_engineering():
    data = visualise_data()
    y_no_count, y_yes_count =data['y'].value_counts()
    y_yes = data[data['y'] == 'yes']
    y_no = data[data['y'] == 'no']
    y_yes_over = y_yes.sample(y_no_count,replace=True)
    df_balanced = pd.concat([y_yes_over,y_no], axis=0)
    df_balanced['y'].groupby(df_balanced['y']).count()
    df2=df_balanced.copy()
    df2.replace({'job':{'management':0,'technician':1,'entrepreneur':2,'blue-collar':3,'unknown':4,'retired':5,'admin.':6,'services':7,'self-employed':8,'unemployed':9,'housemaid':10,'student':11}},inplace=True)
    df2.replace({'marital':{'single':0,'married':1,'divorced':2,'unknown':3}},inplace=True)
    df2.replace({'education':{'primary':0,'secondary':1,'tertiary':2,'unknown':3}},inplace=True)
    df2.replace({'contact':{'cellular':0,'telephone':1,'unknown':2}},inplace=True)
    df2.replace({'month':{'jan':0,'feb':1,'mar':2,'apr':3,'may':4,'jun':5,'jul':6,'aug':7,'sep':8,'oct':9,'nov':10,'dec':11}},inplace=True)
    df2.replace({'poutcome':{'failure':0,'success':1,'unknown':2,'other':3}},inplace=True)
    bool_columns = ['housing', 'loan', 'y']
    for col in  bool_columns:
        df2[col+'_new']=df2[col].apply(lambda x : 1 if x == 'yes' else 0)
        df2.drop(col, axis=1, inplace=True)
    
    #defaut features does not play imp role
    df2.groupby(['y_new','default']).size()
    df2.drop(['default'],axis=1, inplace=True)
    df2.groupby(['y_new','pdays']).size()
    df2.drop(['pdays'],axis=1, inplace=True)
    # remove outliers in feature age...
    df2.groupby('age',sort=True)['age'].count()
    # remove outliers in feature balance...
    df2.groupby(['y_new','balance'],sort=True)['balance'].count()
    # these outlier should not be remove as balance goes high, client show interest on deposit
    # remove outliers in feature campaign...
    df2.groupby(['y_new','campaign'],sort=True)['campaign'].count()
    df3 = df2[df2['campaign'] < 40]
    df3.groupby(['y_new','campaign'],sort=True)['campaign'].count()
    df3.groupby(['y_new','previous'],sort=True)['previous'].count()
    df4 = df3[df3['previous'] < 50]
    df4.groupby(['y_new','previous'],sort=True)['previous'].count()
    # cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    # for col in  cat_columns:
    #     df4 = pd.concat([df4.drop(col, axis=1),pd.get_dummies(df4[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)
    
    dataset = df4.to_csv('cleaned_data.csv',index=False)
    return dataset

feature_engineering()
