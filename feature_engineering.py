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
    #defaut features does not play imp role
    df2.groupby(['y','default']).size()
    df2.drop(['default'],axis=1, inplace=True)
    df2.groupby(['y','pdays']).size()
    df2.drop(['pdays'],axis=1, inplace=True)
    # remove outliers in feature age...
    df2.groupby('age',sort=True)['age'].count()
    # remove outliers in feature balance...
    df2.groupby(['y','balance'],sort=True)['balance'].count()
    # these outlier should not be remove as balance goes high, client show interest on deposit
    # remove outliers in feature campaign...
    df2.groupby(['y','campaign'],sort=True)['campaign'].count()
    df3 = df2[df2['campaign'] < 40]
    df3.groupby(['y','campaign'],sort=True)['campaign'].count()
    df3.groupby(['y','previous'],sort=True)['previous'].count()
    df4 = df3[df3['previous'] < 50]
    df3.groupby(['y','previous'],sort=True)['previous'].count()
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    for col in  cat_columns:
        df4 = pd.concat([df4.drop(col, axis=1),pd.get_dummies(df4[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)
    bool_columns = ['housing', 'loan', 'y']
    for col in  bool_columns:
        df4[col+'_new']=df4[col].apply(lambda x : 1 if x == 'yes' else 0)
        df4.drop(col, axis=1, inplace=True)
    dataset = df4.to_csv('bank_term_deposit_prediction_clean_data.csv')
    return dataset

feature_engineering()
