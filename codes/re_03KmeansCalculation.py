#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:52:26 2024

@author: L.J.Wang
"""

import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def kmeans(data, k, initial_centroids):
    centroids = np.array(initial_centroids)
    while True:
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return clusters

# 11个点的数据
data = np.array([(8,9), (6,8), (1,6), (7,0), (1,1), (4,7), (3,8), (5,5), (7,2), (4,8), (5,6)])

# 初始聚类中心
initial_centroids = [(7,2), (4,8), (5,6)]

# 聚类数量
k = 3

# 进行K-means聚类
clusters = kmeans(data, k, initial_centroids)

# 打印每个类别的数据
for i, cluster in enumerate(clusters):
    print(f'Cluster {i+1}: {cluster}')
