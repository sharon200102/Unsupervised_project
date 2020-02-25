import dataframe as df
import pandas as pd
import numpy as np
data = pd.read_csv('creditcard.csv')


# the function gets vector that present a deal and return 1 if its deception and 0 if its not
def is_decption(vec):
    if vec["Class"] == 0:
        return 0
    else:
        return 1


# the function gets a dataset and return the percent of the clean deals and the deceptionds
def percent(dataset):
    deceptions = 0
    clean_deals = 0
    for i in range(len(dataset["Class"])):
        if is_decption(dataset.loc[i]) == 0:
            clean_deals = clean_deals + 1
        else:
            deceptions = deceptions + 1
    return [clean_deals / len(dataset["Class"]), deceptions/ len(dataset["Class"])]


# the function gets a dataset and return his covarince
def all_cov(dataset):
    return dataset.cov()


# the function gets a dataset and return array with the covariance between the amount column and the all others
def amount_cov(dataset):
    arr = []
    for coll in dataset.columns:
        d = {"Amount": dataset["Amount"], coll: dataset[coll]}
        df = pd.DataFrame(data=d)
        arr.append(df.cov())
    return arr




