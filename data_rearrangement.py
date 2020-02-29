import pandas as pd
import numpy as np
def minmax_norm(df):
    return (df-df.min())/(df.max()-df.min())
def zscore_norm(df):
    return (df - df.mean()) / df.std()
