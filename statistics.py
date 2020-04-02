import pandas as pd
import numpy as np
from scipy import stats
import load_data as ld
import seaborn as sns
import matplotlib.pyplot as plt
THRESHOLD=0.05
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


def corr(data):
    df = data.corr()
    return df.min(), df[df < 1].max()
# The function performs a T test between  two dataframes (column by column) and returns a list of results.
def TtestDf(df1,df2):
    columns=df1.columns
    l=[]
    #Iterate through all columns.
    for col in columns:
        l.append(stats.ttest_ind(df1[col],df2[col], equal_var=False)[1])
    return l
"""----T_test----"""
"""
fraud=ld.data[ld.class_col==1]
legal=ld.data[ld.class_col==0]
p_value_list=TtestDf(fraud,legal)
print(p_value_list)
fig,ax=plt.subplots(figsize=(20,20))
sns.barplot(ld.data.columns,p_value_list,ax=ax)
ax.axhline(y=THRESHOLD,linewidth=1, color='r',ls='--',label='P = 0.05')
ax.set_title('P values of T test between columns of frauds and legal transactions \n Some bars are missing due to a '
             'very small P value') 
ax.set_ylabel('P values')
ax.set_xlabel('Column names')
plt.legend()
plt.tight_layout()
plt.show()
"""


