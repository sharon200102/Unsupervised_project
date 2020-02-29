import pandas as pd
import numpy as np
def minmax_norm(df):
    return (df-df.min())/(df.max()-df.min())
def zscore_norm(df):
    return (df - df.mean()) / df.std()
data=pd.read_csv(r'C:\sharon\second_degree\unsupervised_learning\Final Project\Unsupervised_project\creditcard.csv')
data.drop('Class',inplace=True,axis=1)
zscore_norm(data).to_csv('MinMax_normalized_data.csv')