from sklearn.cluster import KMeans ,DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import decomposition

import numpy as np
import load_data as ld

figsize=(15,15)
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

kmeans = KMeans(n_clusters=n_clusters)
gmm = GaussianMixture(n_components=n_clusters)
clustring_algorithms=(('Kmeans',kmeans),('GMM',gmm))
normalization=[('MiniMax',ld.minmax_norm),('Zscore',ld.zscore_norm),('RobustScaler',ld.Robust_std)]
decomposition_algorithms=[('PCA', PCA), ('ICA', FastICA)]
fig,axes=plt.subplots(nrows=len(clustring_algorithms)*len(normalization)*len(decomposition_algorithms),ncols=1,figsize=figsize)
i=0

for norm_name, norm in normalization:
    normalized_data=norm(data)
    for dec_name,dec in decomposition_algorithms:
        decomposed_data=decomposition.decompose(normalized_data,dec,n_components)[1]
        for alg_name,alg in clustring_algorithms:
            alg.fit(decomposed_data)
            if hasattr(alg, 'labels_'):
                y_pred = alg.labels_.astype(np.int)
            else:
                y_pred = alg.predict(decomposed_data)
            create_silhouette(decomposed_data,y_pred,n_clusters,axes[i])
            axes[i].set_title("Silhouette analysis of "+norm_name+" - "+dec_name+" - "+alg_name)
            i+=1

plt.tight_layout()
plt.show()




