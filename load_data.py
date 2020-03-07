import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

data = pd.read_csv('creditcard.csv')


def minmax_norm(df):
    return (df-df.min())/(df.max()-df.min())


def zscore_norm(df):
    return (df - df.mean()) / df.std()

def Robust_std(df):
    transformer = RobustScaler().fit(df)
    return pd.DataFrame(transformer.transform(df),columns=df.columns)

def kick_anom(dataset, coll_name, limit):
    for i in range(len(dataset[coll_name])):
        if dataset.loc[i][coll_name] > limit:
            dataset = dataset.drop(i)
    return dataset
class_col=data['Class']
data = data.drop(columns=["Class"])
data = minmax_norm(data)
