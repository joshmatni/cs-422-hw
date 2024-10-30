# Joshua Matni
# CS 422 Project 2
# KMEANS

'''
Implement the K-Means Clustering algorithm in python. You should implement the two functions listed. 
K refers to the number of random points to act as cluster centers. X refers to the data being passed 
(where the data only consists of 2 features of your choosing). The function should return the generated 
clusters. The returned clusters should then be passed to the plot function, which will generate plots using 
matplotlib. Make sure to color code your clusters for visibility. The plot function does not need a return. 
You will be graded on this portion based off the graphs you generate. The Iris dataset is provided for testing. 
You will want to output multiple plots where the x and y axes use different combinations of the features provided 
in the dataset. It is expected to try and use multiple values of K to find which parameter produces the most useful plots.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# K refers to the number of random points to act as cluster centers
# X refers to the data being passed 
def kmeans(k, X):
    # select random data points for clusters
    clusters = X[np.random.choice(X.shape[0], size=k, replace=False)]
    not_converged = True
    
    while not_converged:
        # assign data to clusters
        cluster_assignments = []
        for i in range(len(X)):
            distances = []
            for j in range(len(clusters)):
                distance = np.linalg.norm(X[i] - clusters[j])
                distances.append(distance)
                
            # get min index of distances
            min_dist_index = np.argmin(distances)
            cluster_assignments.append(min_dist_index)
        
        # update centroids
        new_centroids = []
        for i in range(len(clusters)):
            points_in_cluster_i = [X[j] for j in range(len(X)) if cluster_assignments[j] == i]
            if points_in_cluster_i: # check if a cluster has a point
                new_centroid = np.mean(points_in_cluster_i, axis=0) # averaging across the rows, which gives new centroid with average value for each feature
            else:
                # if cluster empty, keep the centroid unchanged
                new_centroid = clusters[i]
        
            new_centroids.append(new_centroid)

        # check if converged
        if np.allclose(clusters, new_centroids, atol=1e-5):
            not_converged = False

        clusters = new_centroids


    return cluster_assignments, np.array(clusters)

def plot(clusters, X, feature_names):
    cluster_assignments, centroids = clusters
    
    # list of feature combinations to plot ([(0,1), (0,2), (1,3)] means Sepal Length vs Petal Width, etc
    feature_combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    for x_idx, y_idx in feature_combinations:
        plt.figure()  # figure for each plot
        
        # color by cluster assignment
        plt.scatter(X[:, x_idx], X[:, y_idx], c=cluster_assignments, cmap='viridis', marker='o')
        plt.scatter(centroids[:, x_idx], centroids[:, y_idx], s=300, c='red', marker='X', label='Centroids')
        
        plt.xlabel(feature_names[x_idx])
        plt.ylabel(feature_names[y_idx])
        plt.title(f'K-Means Clustering: {feature_names[x_idx]} vs {feature_names[y_idx]}')
        plt.legend()
        plt.show()


def main():
    df = pd.read_csv("iris.csv")
    
    X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
    feature_names = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    
    k = 4
    cluster_assignments, centroids = kmeans(k, X)
    clusters = (cluster_assignments, centroids)
    
    plot(clusters, X, feature_names)


if __name__ == "__main__":
    main()