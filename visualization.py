import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
dataset = pd.read_csv('creditcard.csv')


# Plots the distribution of all columns by the shape and size inserted as arguments.

def dist_of_cols(col_list,data,figsize,nrows=1,ncols=1,savefig=0,name_of_file=None):
    idx=0
    if len(col_list)!=nrows*ncols:
      raise ValueError("The number of rows an columns inserted doesn't match to the length of col_list argument")
    fig, axes=plt.subplots(nrows=nrows,ncols=ncols,squeeze=False,figsize=figsize)
    # run over all lines in the grid.
    for ax_list in axes:
    # run over all axes in the line.
      for ax in ax_list:
        ax.set_title(col_list[idx]+" disribution")
        #plot the distribution of the column.
        sns.distplot(data[col_list[idx]],ax=ax)
        idx+=1
    # If the user wants to save the figure.
    if(savefig==1):
      fig.savefig(name_of_file)
    plt.tight_layout() #add spacing to the figure.
    plt.show()
#Plots the different distributions of a column inserted based on a categorical feature.
def dis_by_cat(data,col_name,cat_name,figsize,savefig=0,name_of_file=None):
  g = sns.FacetGrid(data=data,row=cat_name,height=figsize[0],aspect=figsize[1],sharex=False)
  g = g.map(sns.distplot, col_name)
  # If the user wants to save the figure.
  if(savefig==1):
      g.savefig(name_of_file)
  plt.tight_layout()
  plt.show()
def scatterdDfVisualization(df,fig,**kwargs):
    n_components=df.shape[1]
    if n_components == 3:
        ax = Axes3D(fig)
        ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],**kwargs)
    if n_components == 2:
        ax = fig.add_subplot()
        sns.scatterplot(df.iloc[:,0], df.iloc[:,1], data=df,ax=ax)
    return ax
"""
    sns.barplot(x='Class',y='Amount',data=data )
    sns.stripplot(x="Class", y="Amount", data=data)
"""

"""
making TSNE
"""
def make_data_for_TSNE(data):
    df = data[data['Class'] == 1]
    zf = data[data['Class'] == 0]
    zf = shuffle(zf, random_state=1)
    s = len(df)
    zf = zf.head(s)
    frames = [df, zf]
    new_data = pd.concat(frames)
    new_data = shuffle(new_data, random_state=1)
    return new_data


def TSNE_analysis(data, n_components=2, savefig=0, name_of_file=None):
    tsne_data = make_data_for_TSNE(data)
    tsne = TSNE(n_components=n_components, perplexity=3)
    tsne.fit(tsne_data)
    principalComponents = tsne.fit_transform(tsne_data)
    principalDf = pd.DataFrame(data=principalComponents, columns=list(map(lambda num : 'principal component '+str(num), range(1, n_components+1))))
    if savefig == 1:
        fig = plt.figure()
        ax = scatterdDfVisualization(principalDf, fig, c=tsne_data['Class'])
        ax.set_xlabel('First component')
        ax.set_ylabel('Second component')
        ax.set_zlabel('Third component')
        ax.set_title('3D TSNE plot')
        plt.show()
        fig.savefig(name_of_file)
    return tsne, principalDf

"""
TSNE_analysis(dataset, n_components=3, savefig=1, name_of_file="3D_TSNE_analysis_plot")
"""
