import pandas as pd
import numpy as np
import load_data as ld
from visualization import scatterdDfVisualization
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
# function for decomposing the data to fewer dimensions using the constructor function inserted.
def decompose(data,dec_func,n_components=2):
    dec_obj = dec_func(n_components=n_components)
    dec_obj.fit(data)
    components = dec_obj.fit_transform(data)
    #changing the columns names.
    componentsDF = pd.DataFrame(data=components, columns=list(map(lambda num:'component '+str(num),range(1,n_components+1))))
    return dec_obj,componentsDF
#The function returns the most importent fetures in the orignal_data
def importent_feat(dec_obj,orignal_data_columns):
    n_components = dec_obj.components_.shape[0]
    most_important_feat = [np.abs(dec_obj.components_[i]).argmax() for i in range(n_components)]
    important_feat_names = [orignal_data_columns[index] for index in most_important_feat]
    return  important_feat_names
"""
fig=plt.figure()
normalized_data=ld.minmax_norm(ld.data)
componentsDF=decompose(normalized_data,FastICA,n_components=3)[1]
ax=scatterdDfVisualization(componentsDF,fig,c=ld.class_col)
ax.set_xlabel('First component')
ax.set_ylabel('Second component')
ax.set_zlabel('Third component')
ax.set_title('3D ICA by class analysis')
plt.show()
"""
