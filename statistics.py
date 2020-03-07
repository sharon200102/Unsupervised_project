import dataframe as df
import pandas as pd
import numpy as np


# the function gets vector that present a deal and a name of a column and return 1 if its deception and 0 if its not
def is_decption(vec, name):
    if vec[name] == 0:
        return 0
    else:
        return 1


# the function gets a dataset and a name of a column return the percent of the clean deals and the deceptionds
def percent(dataset, name):
    deceptions = 0
    clean_deals = 0
    for i in range(len(dataset[name])):
        if is_decption(dataset.loc[i], name) == 0:
            clean_deals = clean_deals + 1
        else:
            deceptions = deceptions + 1
    return [clean_deals / len(dataset[name]), deceptions/ len(dataset["Class"])]


# the function gets a dataset and return his covarince
def all_cov(dataset):
    return dataset.cov()


# the function gets a dataset and the name of a column return array with the covariance between the name (amount) column and the all others
def amount_cov(dataset, name):
    arr = []
    for coll in dataset.columns:
        d = {name: dataset[name], coll: dataset[coll]}
        df = pd.DataFrame(data=d)
        arr.append(df.cov())
    return arr


def histogram_intersection(a, b):
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


df = data.corr(method=histogram_intersection)
