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


data=ld.data
normalized_data=ld.minmax_norm(data)
vector_size=len(ld.data.columns)
lr=0.001
n_epochs=7
n_cluster=3
class autoencoder(nn.Module):
    def __init__(self, vector_size):
        super(autoencoder, self).__init__()
        """the first part of the auto-encoder two layers, and a relu function between them"""
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


model=autoencoder(vector_size)
criterion = nn.MSELoss()
tensor_data=torch.from_numpy(normalized_data.to_numpy()).float()

train_step=make_train_step(model,criterion,optim.Adam(model.parameters(), lr=lr))
losses = []
"""Training the model"""
for epoch in range(n_epochs):
    print("in epoch "+str(epoch))
    for x in tensor_data:
        losses.append(train_step(x,x))
    fitkmeans(losses,3)
    losses=[]
model.eval()
"""plot a graph also after training"""
for x in tensor_data:
    losses.append(criterion(model(x),x).item())
fitkmeans(losses,3)
#set title !!
#sns.stripplot(x =ld.class_col,y=losses)
plt.show()







