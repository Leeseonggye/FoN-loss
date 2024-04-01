import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import cdist
import numpy as np
import pdb

def get_k_means_cluster_pos_dist(data, metric,n_cluster):
    """
    data: numpy array
    metric: Euclidean or dtw
    n_cluster: # of cluster
    """
    
    # Scale the time series data
    scaler = TimeSeriesScalerMeanVariance()
    scaled_data = scaler.fit_transform(data)

    # Instantiate the K-means clustering algorithm
    kmeans = TimeSeriesKMeans(n_clusters = n_cluster, metric=metric)
    labels = kmeans.fit_predict(scaled_data)
    
    # Compute cluster centroids
    cluster_centroids = kmeans.cluster_centers_
    cluster_distances = cdist(cluster_centroids.reshape(n_cluster, -1), cluster_centroids.reshape(n_cluster, -1), metric='euclidean')
    
    # Average scaling
    sum = 0
    for i in range(len(cluster_distances)):
        for j in range(len(cluster_distances[0])):
            if cluster_distances[i][j] != 0:
                sum = sum + 1/cluster_distances[i][j]

    for i in range(len(cluster_distances)):
        for j in range(len(cluster_distances[0])):
            if cluster_distances[i][j] != 0:
                cluster_distances[i][j] = 1/cluster_distances[i][j]/sum * len(cluster_distances) * (len(cluster_distances)-1)
            
            else:
                cluster_distances[i][j] = 1
    
    loss_weight_matrix = cluster_distances
    

    return labels, loss_weight_matrix