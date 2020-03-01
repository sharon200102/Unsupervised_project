import dataframe as df
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import load_data as ld
pca = PCA(n_components=3)
pca.fit(ld.data)
principalComponents = pca.fit_transform(ld.data)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])
print(principalDf)