#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 23:50:54 2020

@author: Veeru_Mac
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans



def readData():
    df = pd.read_csv("ClusterPlot.csv")
    df.plot(kind='scatter',x='V1',y='V2',color='blue')
    plt.show()
    return df.drop(df.columns[0], axis=1)


def elbowMethod(data):
    #Transforming data values in range of 0<=x<=1 , 0<=y<=1
    mms=MinMaxScaler()
    mms.fit(data)
    data_transformed=mms.transform(data)

    #Converting  data_transformed variable to pandas dataframe
    data_transformed=pd.DataFrame(data_transformed, columns=['V1', 'V2'])
    
    #Elbow method to minimize WSS (Within-cluster Sum of Square)
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km.fit(data_transformed)
        Sum_of_squared_distances.append(km.inertia_)
        
        
    #Plotting the Elbow curve
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    

def getColor(i):
    if(i == 0):
        return 'red'
    elif(i==1):
        return 'green'
    else:
        return 'yellow'

def clustering(data):
    number_of_clusters=3
    
    kmeans = KMeans(n_clusters=number_of_clusters) #defining number of clusters
    kmeans.fit(data)  #fit kmeans object to dataset
    
    #Identifying the clusters
    clusters = kmeans.cluster_centers_
    
    #positions hold values of either 0,1,2 for each point in graph. 0,1,2 corresponds to their respective centroid.
    #These values are used to determine which point belong to which cluster.
    positions = kmeans.fit_predict(data)
    print(positions.shape)

    
    for i in range(number_of_clusters):
        plt.scatter(data.values[positions == i,0], data.values[positions == i,1], s=20, color=getColor(i))
        plt.scatter(clusters[i][0], clusters[i][1], marker='o', s=100, color='black')
        
    plt.show()

        
    

if __name__== "__main__":
    data = readData()
    elbowMethod(data)
    print("Judging by the location of the 'bend' in the graph, a good choice for number of clusters is 3")
    clustering(data)


    














