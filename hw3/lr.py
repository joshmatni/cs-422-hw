# Joshua Matni
# CS 422 project 3
# Linear Regression

"""
Implement a linear regression algorithm using the class template provided above. 
The salary data.csv is provided for testing. It is up to you to define how the algorithm 
performs the fit, update weights, and predict functions. Use the Mean Squared Error formula 
for determining accuracy of the final model. Use gradient descent for updating the weights. 
If not using train, test, split, then simply use randomly sampled values from the whole dataset 
to report on the accuracy. If using train, test, split, then use the testing dataset. Make sure 
to indicate if you used the sklearn module in your report.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate 
        self.iterations = iterations
        #self.weights = self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(self.iterations): 
            self.update_weights(X, y)
        return self
    
    def update_weights(self, X, y):
        predicton = np.dot(X, self.weights) + self.bias
        errors = predicton - y

        # calculate the gradient of the MSE with respect to each weight
        # take partial deriv of loss func X.T (transpose)
        weight_gradient = (2 / X.shape[0]) * np.dot(X.T, errors)
        bias_gradient = (2 / X.shape[0]) * sum(errors)

        # update weights and biases
        self.weights -= (self.learning_rate * weight_gradient)
        self.bias -= (self.learning_rate * bias_gradient)

        return self
    
    def predict(self , X):
        return np.dot(X, self.weights) + self.bias


def main():
    df = pd.read_csv("salary_data.csv")
    X = df["YearsExperience"].values.reshape(-1, 1)
    y = df["Salary"].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lm = LinearRegression(learning_rate=0.01, iterations=1000)
    lm.fit(X_train, y_train)
    
    predicted_value = lm.predict(X_test)
    mse = mean_squared_error(y_test, predicted_value)
    print(f"Predicted Values: {predicted_value}")
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()