import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import load_data as ld
import seaborn as sns
def pca_analysis(data,n_components=2,savefig=0,name_of_file=None):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data=principalComponents, columns=list(map(lambda num:'principal component '+str(num),range(1,n_components+1))))
    if savefig==1:
        fig = plt.figure()
        if n_components==3:
            ax=Axes3D(fig)
            ax.scatter(principalDf['principal component 1'],principalDf['principal component 2'],principalDf['principal component 3'])
            fig.savefig(name_of_file)
        if n_components==2:
            ax=fig.add_axes()
            sns.scatterplot(principalDf['principal component 1'],principalDf['principal component 2'],data=data,ax=ax)
            fig.savefig(name_of_file)
    return pca,principalDf