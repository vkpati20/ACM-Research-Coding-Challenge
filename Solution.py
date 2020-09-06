#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 23:50:54 2020

@author: Veerendranath Korrapati
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np


#To read data and return dataset columns as Pandas DataFrame
def readData():
    df = pd.read_csv("ClusterPlot.csv")
    df.plot(kind='scatter',x='V1',y='V2',color='blue')
    plt.show()
    return df.drop(df.columns[0], axis=1)


#To identify number of clusters
def elbowMethod(data):    
    distortions = []
    K = range(1,10)
    for k in K:
        model = KMeans(n_clusters=k).fit(data)
        model.fit(data)
        distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    
    #Plotting the Elbow curve
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortions')
    plt.title('Elbow Method For Optimal k')
    plt.show()



def getColor(i):
    if(i == 0):
        return 'red'
    elif(i==1):
        return 'green'
    else:
        return 'yellow'

#To classify points based on number of clusters
def clustering(data):
    print("Judging by the location of the 'bend' in the graph, a good choice for number of clusters is 3")
    number_of_clusters=3
    
    kmeans = KMeans(n_clusters=number_of_clusters) 
    kmeans.fit(data)
    
    #Identifying the centroids
    centroids = kmeans.cluster_centers_
    
    #positions hold values of either 0,1,2 for each point in graph. 0,1,2 corresponds to their respective centroid.
    #These values are used to determine which point belong to which cluster.
    positions = kmeans.fit_predict(data)

    
    for i in range(number_of_clusters):
        plt.scatter(data.values[positions == i,0], data.values[positions == i,1], s=20, color=getColor(i))
        plt.scatter(centroids[i][0], centroids[i][1], marker='o', s=100, color='black')
        
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.title('Clustering')
    plt.show()

        
    

if __name__== "__main__":
    data = readData()
    elbowMethod(data)
    clustering(data)


    














