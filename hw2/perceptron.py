# Josh Matni
# CS 422 Project 2
# PERCEPTRONNN

'''
Implement the Perceptron algorithm in python. The above function should accept data samples (X), labels (y), 
and a smaller subset of data samples for testing the performance of the model. The function should return the 
accuracy of the model. The Iris dataset is provided for testing. Using the clusters from the K-means portion 
of this project, pick only two of the classes to work with (perceptron’s can only do binary classification) to 
implement your solution. The terminal should output the accuracy of the perceptron. You will be graded based off 
of a smaller subset of the Iris dataset passed into the x test parameter.
'''

import pandas as pd
import numpy as np

def perceptron(X, y, x_test) -> float:
    # initial weights for each feature and bias
    w_sepal_length, w_sepal_width, w_petal_length, w_petal_width, bias = 0, 0, 0, 0, 0
    X = np.array(X)
    y = np.array(y)

    # 100 epochs
    for i in range(0, 100):
        for j in range(len(X)):  # loop over each data point
            # extract all four features from the current data point
            sepal_length, sepal_width, petal_length, petal_width = X[j]
            label = y[j]
            learning_rate = 0.1

            # calculate weighted sum with all features
            weighted_sum = (
                (w_sepal_length * sepal_length) +
                (w_sepal_width * sepal_width) +
                (w_petal_length * petal_length) +
                (w_petal_width * petal_width) +
                bias
            )

            # activation function
            prediction = 1 if weighted_sum > 0 else -1

            # compare prediction to label
            if prediction != label:
                error = label - prediction

                # update each weight and bias
                w_sepal_length += learning_rate * error * sepal_length
                w_sepal_width += learning_rate * error * sepal_width
                w_petal_length += learning_rate * error * petal_length
                w_petal_width += learning_rate * error * petal_width
                bias += learning_rate * error

    # evaluate how well the model performs on original training set
    correct_predictions = 0
    for i in range(len(X)):
        # Extract all four features
        sepal_length, sepal_width, petal_length, petal_width = X[i]
        label = y[i]

        # calculate weighted sum with the final weights and bias
        weighted_sum = (
            (w_sepal_length * sepal_length) +
            (w_sepal_width * sepal_width) +
            (w_petal_length * petal_length) +
            (w_petal_width * petal_width) +
            bias
        )
        prediction = 1 if weighted_sum > 0 else -1

        if prediction == label:
            correct_predictions += 1

    # calculate accuracy
    accuracy = (correct_predictions / len(X)) * 100

    # output predicted labels for x_test
    x_test_predictions = []
    for test_point in x_test:
        sepal_length, sepal_width, petal_length, petal_width = test_point
        weighted_sum = (
            (w_sepal_length * sepal_length) +
            (w_sepal_width * sepal_width) +
            (w_petal_length * petal_length) +
            (w_petal_width * petal_width) +
            bias
        )
        prediction = 1 if weighted_sum > 0 else -1
        x_test_predictions.append(prediction)
    
    print("Predicted labels for x_test:", x_test_predictions)

    return accuracy


def main():
    df = pd.read_csv("iris.csv")
    df = df[df["variety"].isin(["Setosa", "Versicolor"])]  # filter for 2 classes only

    # map classes to numeric labels where Setosa = 1, Versicolor = -1
    df["variety"] = df["variety"].map({"Setosa": 1, "Versicolor": -1})

    X = df[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
    y = df["variety"]

    # few samples for x_test predictions
    x_test = X.iloc[-5:].values  # last 5 samples as x_test
    X = X.iloc[:-5]  # exclude x_test samples from training data
    y = y.iloc[:-5]

    accuracy = perceptron(X, y, x_test)
    print(f"Perceptron training accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()