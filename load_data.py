import pandas as pd
import numpy as np
data = pd.read_csv('creditcard.csv')


def minmax_norm(df):
    return (df-df.min())/(df.max()-df.min())


def zscore_norm(df):
    return (df - df.mean()) / df.std()


def kick_anom(dataset, coll_name, limit):
    for i in range(len(dataset[coll_name])):
        if dataset.loc[i][coll_name] > limit:
            dataset = dataset.drop(i)
    return dataset


data = data.drop(columns=["Class"])
data = minmax_norm(data)
