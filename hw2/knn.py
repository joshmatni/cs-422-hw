# Josh Matni
# CS 422 Project 2
# K-Nearest Neighbors

'''
Implement the K-Nearest Neighbor algorithm in python with the above function as the algorithm tem-plate.
K refers to the value of amount of nearest neighbors to train with, while X and y refer to the data samples 
and their labels to train the model on. x test should accept some other data samples to test the performance 
of the model. For grading, your code will be tested against a subset of the iris dataset for different values of k. 
The Iris dataset is provided for testing. The function should return the accuracy of the model, which should be 
output to the terminal.
'''

import pandas as pd
import numpy as np

def knn(k, X, y, x_test):
    # b/c x_test is not used in the accuracy calculation; perform leave-one-out cross-validation on X and y
    predictions = []
    X = np.array(X) 
    y = np.array(y)
    
    for i in range(len(X)):
        # training data excluding the current sample
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        
        # current sample to test
        x_current, y_current = X[i], y[i]
        
        # euclidean distance from x_current to all points in X_train
        distances = np.linalg.norm(X_train - x_current, axis=1)
        neighbor_indices = distances.argsort()[:k] # returns sorted knn indicies 
        
        # grab labels
        neighbor_labels = y_train[neighbor_indices]
        
        # predict the label by majority vote
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        most_frequent_label = labels[counts.argmax()]
        predictions.append(most_frequent_label)
    
    correct_predictions = sum(pred == actual for pred, actual in zip(predictions, y))
    accuracy = correct_predictions / len(y)
    
    # output predicted labels for x_test
    test_predictions = []
    for x in x_test:
        distances = np.linalg.norm(X - x, axis=1)

        neighbor_indices = distances.argsort()[:k]
        neighbor_labels = y[neighbor_indices]
        
        labels, counts = np.unique(neighbor_labels, return_counts=True)
        most_frequent_label = labels[counts.argmax()]
        test_predictions.append(most_frequent_label)

    print("Predictions for x_test:", test_predictions)
    return accuracy

def main():
    df = pd.read_csv('iris.csv')

    X = df.iloc[:, 0:4].values  # features
    y = df.iloc[:, 4].values    # labels

    x_test = X[:5]
    k = 5
    accuracy = knn(k, X, y, x_test)
    print(f"Accuracy with k={k}: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
