import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle



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
# dic should look as follows {class_num:desired quantity of rows from that specified class}
# The dictionary returned will look as follows {class_num: dataframe with quantity inserted of rows of class_num}
def divide_data(data,labels,dic):
    outdic={}
    for key in dic:
        key_class_df=shuffle(data[labels==key],random_state=1)
        outdic[key]=key_class_df.head(dic[key])

    return outdic
#dic should look as follows,{class_num:dataframe}
# returns a concatenate dataframe of all values in dic and a corresponding labels
def concatenate_data(dic):
    labels=[]
    for key in dic:
        labels+=len(dic[key])*[key]
    return pd.concat(dic.values()),np.array(labels)



original_data = pd.read_csv('creditcard.csv')
class_col=original_data['Class']
data = original_data.drop(columns=["Class"])
