import torch
import torch.nn as nn
from torch import optim
from heapq import nlargest
import pandas as pd
from numpy import linspace,array
import seaborn as sns
import load_data as ld
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix


normalized_data=ld.zscore_norm(ld.data)
# Transform the 0-1 labels to 1-(-1) labels because of the Isolation forest
anomaly_labels=ld.class_col.apply(lambda x: -2*x+1)
vector_size=len(ld.data.columns)
lr=0.001
n_epochs=1
# The neural net that will preform the auto-encoding
class autoencoder(nn.Module):
    def __init__(self, vector_size):
        super(autoencoder, self).__init__()
        """The first part of the auto-encoder two layers, and a relu function between them"""
        self.encoder=nn.Sequential(
        nn.Linear(vector_size,6),
        nn.ReLU(True),
        nn.Linear(6,3)
        )
        """Symmetric to the first part"""
        self.decoder=nn.Sequential(
            nn.Linear(3,6),
            nn.ReLU(True),
            nn.Linear(6,vector_size),
            nn.Tanh()
        )
    def forward(self,input):
        input=self.encoder(input)
        input=self.decoder(input)
        return input
    
    def Dimensions(self, input):
        input=self.encoder(input)
        return input


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step
def fitkde(X):
    """fits a Kde to the one dimensional losses"""
    a=array(X).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = linspace(0, max(X))
    e = kde.score_samples(s.reshape(-1, 1))
    plt.plot(s, e)
    plt.show()

def fitkmeans(X,n_clusters=3):
    """fits a kmeans to the one dimensional losses"""
    a=array(X).reshape(-1, 1)
    kmeans=KMeans(n_clusters)
    y_pred=kmeans.fit_predict(a)
    sns.stripplot(y_pred,X,hue=ld.class_col)
    plt.show()

    
def auto_for_dimensions(arr, n_cluster,savefig=0, name_of_fig=None):
    fig_arr = np.zeros((len(ld.data), n_cluster))
    for i in range(len(ld.data)):
        for j in range(n_cluster):
            fig_arr[i][j] = arr[i][j].float()
    if savefig == 1:
        fig = plt.figure()
        ax = Axes3D(fig)
        principalDf = fig_arr
        ax.scatter(principalDf[:, 0], principalDf[:, 1], principalDf[:, 2], c=ld.class_col)
        fig.savefig(name_of_fig)
    return fig_arr


"""
FOR THE ISOLATION FOREST ANOMALY DETECTION
"""
"""
#creating and fitting the Isolation Forest on the data
clf = IsolationForest(random_state=1)
y_pred=clf.fit_predict(normalized_data)
cm=confusion_matrix(anomaly_labels,y_pred)
cm_df=pd.DataFrame(cm,index=['Fraud','Legal'],columns=['Predicted_fraud','Predicted_legal'])

recall = cm[0, 0] / sum(cm[0, :])
#Confusion Matrix
ax=sns.heatmap(cm_df, annot=True, fmt='g',cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix of Isolation Forest')
plt.xlabel('The fraud recall is: '+str(recall)[:4])
plt.show()
"""











"""
FOR THE AUTO-ENCODER ANOMALY DETECTION
"""

"""
# creating the auto-encoder and transforming the data to tensor format
model=autoencoder(vector_size)
criterion = nn.MSELoss()
tensor_data=torch.from_numpy(normalized_data.to_numpy()).float()

train_step=make_train_step(model,criterion,optim.Adam(model.parameters(), lr=lr))
losses = []
#training the model on the whole data.
for epoch in range(n_epochs):
    new_arr = []
    print("in epoch "+str(epoch))
    for x in tensor_data:
        train_step(x,x)
        new_arr.append(model.Dimensions(x))
model.eval()
for x in tensor_data:
    losses.append(criterion(model(x),x).item())
Selecte ten percent of the data with the highest loss as anomalies 
anomaly_index=np.argsort(losses)[-1*int(len(tensor_data)/10):]
print(-1*int(len(tensor_data)/10))
y_pred=np.ones(len(tensor_data))
y_pred[anomaly_index]=-1
print(y_pred)

# confusion_matrix
cm=confusion_matrix(anomaly_labels,y_pred)
cm_df=pd.DataFrame(cm,index=['Fraud','Legal'],columns=['Predicted_fraud','Predicted_legal'])
recall = cm[0, 0] / sum(cm[0, :])
ax=sns.heatmap(cm_df, annot=True, fmt='g',cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix of Auto-encoder')
plt.xlabel('The fraud recall is: '+str(recall)[:4])
plt.show()
"""

"""
Plots of graphs
"""
"""
sns.barplot(x=ld.class_col,y=losses)
plt.title('Auto-encoder-anomaly-detection-Average-loss')
plt.ylabel('Average loss')
plt.show()
print(auto_for_dimensions(new_arr, n_cluster, savefig=1, name_of_fig="auto encoder"))
"""






