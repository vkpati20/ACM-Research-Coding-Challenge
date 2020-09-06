# ACM Research Coding Challenge (Fall 2020)

## Libraries
<strong>import pandas</strong>: To read CSV data.<br>
<strong>import matplotlib.pyplot</strong>: To graph results.<br>
<strong>from scipy.spatial.distance import cdist</strong>: To compute Euclidean distance between points to classify a point into a cluster.<br>
<strong>from sklearn.cluster import KMeans</strong>: To partitioning given points into a set of k groups.<br>
<strong>import numpy as np</strong>: To work with arrays in Python.<br>

## Functions
### readData()
Reads data from CSV file, plots current dataset, returns V1 and V2 column set as new Pandas DataFrame
<br>

### elbowMethod(data)
<br>
Takes Pandas DataFrame, with V1 and V2 columns, as parameter and uses Elbow Method to determine number of clusters<br><br>
For each sample value of k, I'm calculating average of the squared distances from the cluster centers of the respective clusters, storing the results in distortions list.<br>
Then I uses the values in distortion list to identify the 'bend' in the graph to determine the value for number of clusters

### clustering(data)
<br>
Takes Pandas DataFrame, with V1 and V2 columns, as parameter and uses KMeans algorithm to group datapoints based on number of clusters
<br><br>
Algorithm: 
Starts with randomly selected centroids(for each cluster), then performs repetitive calculations to optimize for best positions of the centroid. Then performs calculations to catagorize points in the dataset  into a cluster.

## Results
### Dataset Graph
![](images/dataset.png)

### Elbow Graph
From the graph, we can infer that the 'bend' occurs at k=3, therefore, number of clusters = 3
![](images/elbowGraph.png)

### Cluster Graph
![](images/clustering.png)


## Sources
To Find optimal Clusters (Elbow Method):
https://pythonprogramminglanguage.com/kmeans-elbow-method/
To perform KMeans clustering: 
https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
