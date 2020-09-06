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



df = pd.read_csv("ClusterPlot.csv")



df.plot(kind='scatter',x='V1',y='V2',color='blue')
plt.show()

df = df.drop(df.columns[0], axis=1)
print(df)

mms=MinMaxScaler()
mms.fit(df)
data_transformed=mms.transform(df)

data_transformed=pd.DataFrame(data_transformed, columns=['V1', 'V2'])


data_transformed.plot(kind='scatter', x='V1', y='V2')
plt.show()

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

print("Judging by the location of the 'bend' in the graph, a good choice for number of clusters is 3")

number_of_clusters=3


kmeans = KMeans(n_clusters=number_of_clusters) #defining number of clusters
kmeans.fit(df)  #fit kmeans object to dataset

#Identifying the clusters
clusters = kmeans.cluster_centers_

#Now, we need to readjust the values of clusters until convergence

#trying to recalculate the position of our clusters using our dataset
y_km = kmeans.fit_predict(df)
    #y_km is new object



def getColor(i):
    if(i == 0):
        return 'red'
    elif(i==1):
        return 'green'
    else:
        return 'yellow'



for i in range(number_of_clusters):
    plt.scatter(df.values[y_km == i,0], df.values[y_km == i,1], s=20, color=getColor(i))
    plt.scatter(clusters[i][0], clusters[i][1], marker='o', s=100, color='black')


plt.show()










