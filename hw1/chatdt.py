import numpy as np
import pandas as pd
import sys

def cmd_line_args(arg):
    if arg == '1':
        print(arg)
        data = pd.read_csv('test_data.csv')
        X = data[['feature_1', 'feature_2']].to_numpy()
        y = data['label'].to_numpy()
        return X, y
    elif arg == '2':
        print(arg)
        data = pd.read_csv('contrast_data.csv')
        X = data[['feature_1', 'feature_2']].to_numpy()
        y = data['label'].to_numpy()
        return X, y
    elif arg == '3':
        print(arg)
        data = pd.read_csv('real_data.csv')
        X = data[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].to_numpy()
        y = data['label'].to_numpy()
        return X, y
    else:
        print("Invalid Argument")
        return None, None

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # index of the feature to split on
        self.threshold = threshold    # value to split the feature on
        self.left = left              # left child node (where feature <= threshold)
        self.right = right            # right child node (where feature > threshold)
        self.value = value            # class value for leaf nodes

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        # Add a small epsilon to probabilities to avoid log(0)
        ps = np.where(ps > 0, ps, 1e-10)
        return -np.sum(ps * np.log2(ps))

    def _information_gain(self, X_column, y, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create splits
        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold

        if sum(left_idxs) == 0 or sum(right_idxs) == 0:
            return 0  # No information gain if one side is empty

        # Weighted average of entropy for the two children
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        # Information gain is the reduction in entropy
        ig = parent_entropy - child_entropy
        return ig

    def _best_split(self, X, y, features):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature in features:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _most_common_label(self, y):
        if len(y) == 0:
            return None  # Or return a default label
        else:
            return np.bincount(y).argmax()

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for empty dataset
        if n_samples == 0:
            return None

        # Check for stopping conditions
        if n_labels == 1:
            # All samples have the same label
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        if self.max_depth is not None and depth >= self.max_depth:
            # Reached maximum depth
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        if n_samples < self.min_samples_split:
            # Not enough samples to split further
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Find the best split
        features = np.arange(n_features)
        best_feature, best_threshold = self._best_split(X, y, features)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Create child nodes
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = X[:, best_feature] > best_threshold

        # Check if any split resulted in an empty set
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        # Recurse on left and right splits
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return DecisionTreeNode(best_feature, best_threshold, left, right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _traverse_tree(self, x, node):
        if node is None or node.is_leaf_node():
            return node.value if node else None

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        accuracy = correct / len(y) * 100  # As a percentage
        return accuracy


X, y = cmd_line_args(sys.argv[1])

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
preds = tree.predict(X)

# Calculate accuracy
acc = tree.accuracy(X, y)
print(f"Accuracy: {acc:.2f}%")


# Example usage:
# tree = DecisionTree(max_depth=3)
# tree.fit(X, y)
# preds = tree.predict(X)


# code breaks with contradictory datapoints (two of same sample, but with opposite labels)
# code breaks when setting no max_depth (students should figure out how to terminate at deeper depths)
# no method to measure accuracy
# For grad students (split the into a training and test data set)