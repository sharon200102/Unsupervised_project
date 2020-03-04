import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import load_data as ld
import seaborn as sns
# function for decomposing the data to fewer dimensions using the constructor function inserted.
def decompose(data,dec_func,n_components=2,savefig=0,name_of_file=None):
    dec_obj = dec_func(n_components=n_components)
    dec_obj.fit(data)
    components = dec_obj.fit_transform(data)
    #changing the columns names.
    componentsDF = pd.DataFrame(data=components, columns=list(map(lambda num:'component '+str(num),range(1,n_components+1))))
    #visualization
    if savefig==1:
        fig = plt.figure()
        if n_components==3:
            ax=Axes3D(fig)
            ax.scatter(componentsDF['component 1'],componentsDF['component 2'],componentsDF['component 3'])
            fig.savefig(name_of_file)
        if n_components==2:
            ax=fig.add_axes()
            sns.scatterplot(componentsDF['component 1'],componentsDF['component 2'],data=data,ax=ax)
            fig.savefig(name_of_file)
    return dec_obj,componentsDF

decompose(ld.data,FastICA,n_components=2,savefig=1,name_of_file='2d_ica_plot.png')