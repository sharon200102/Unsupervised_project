from sklearn.cluster import KMeans ,DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import decomposition
import pandas as pd
import numpy as np
import load_data as ld
from sklearn.metrics import confusion_matrix
import seaborn as sns
labels=ld.class_col
figsize=(20,20)
data=ld.data
n_clusters=2
n_components=3
def create_silhouette(data,prediction,n_clusters,ax):
    y_lower = 10
    sample_silhouette_values = silhouette_samples(data, prediction)
    for i in range(n_clusters):
        ith_cluster_silhouette_values =sample_silhouette_values[prediction == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return ax

def cluster_components(y_pred,labels,dic):
    df=pd.DataFrame({'labels':labels,'prediction':y_pred})
    for pred in np.unique(y_pred):
        for label in np.unique(labels):
            label_con=df['labels']==label
            pred_con=df['prediction']==pred
            per=len(df[label_con    &   pred_con])/len(df[pred_con])
            print('The percentage of '+str(dic[label])+' in cluster number '+str(pred)+' is : '+str(per)+'\n')

"""
----silhouette----
"""

"""
kmeans = KMeans(n_clusters=n_clusters)
gmm = GaussianMixture(n_components=n_clusters)
clustring_algorithms=(('Kmeans',kmeans),('GMM',gmm))
normalization=[('MiniMax',ld.minmax_norm),('Zscore',ld.zscore_norm)]
decomposition_algorithms=[('PCA', PCA), ('ICA', FastICA)]
fig_gmm,axes_gmm=plt.subplots(nrows=len(normalization)*len(decomposition_algorithms),ncols=1,figsize=figsize)
fig_kmeans,axes_kmeans=plt.subplots(nrows=len(normalization)*len(decomposition_algorithms),ncols=1,figsize=figsize)
axes_dic={'Kmeans':axes_kmeans,'GMM':axes_gmm}
i=0

for norm_name, norm_fn in normalization:
    normalized_data=norm_fn(data)
    for dec_name,dec in decomposition_algorithms:
        decomposed_data=decomposition.decompose(normalized_data,dec,n_components)[1]
        for alg_name,alg in clustring_algorithms:
            alg.fit(decomposed_data)
            if hasattr(alg, 'labels_'):
                y_pred = alg.labels_.astype(np.int)
            else:
                y_pred = alg.predict(decomposed_data)
            print("For " + norm_name + " - " + dec_name + " - " + alg_name)
            create_silhouette(decomposed_data,y_pred,n_clusters,axes_dic[alg_name][i])
            axes_dic[alg_name][i].set_title("Silhouette analysis of "+norm_name+" - "+dec_name+" - "+alg_name)
            cluster_components(y_pred,labels,{0:'Legal transactions',1:'Fraud transactions'})
            silhouette_avg = silhouette_score(decomposed_data, y_pred)
            print(" The silhouette_score is : "+str(silhouette_avg)+'\n')
        i += 1


plt.tight_layout()
plt.show()
"""
"""
----Confusion matrix----
"""
"""
normalized_data=ld.zscore_norm(data)
decomposed_data=decomposition.decompose(normalized_data,PCA,n_components)[1]
gmm = GaussianMixture(n_components=n_clusters)
y_pred=gmm.fit_predict(decomposed_data)


cm=confusion_matrix(labels,y_pred)
cm_df=pd.DataFrame(cm,index=['Legal','Fraud'],columns=['Cluster 0 ','Cluster 1'])
#Confusion Matrix
ax=sns.heatmap(cm_df, annot=True, fmt='g',cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix of Zscore - PCA - GMM ')
plt.show()
"""



