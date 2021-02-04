#############################################################################
# 60012: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit(), predict() and prune() methods of
# DecisionTreeClassifier. You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import math


class SplitCondition:
    """ Class that stores the split condition (e.g x > num)"""

    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.classes = ["x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
                        "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    def check(self, row_to_check):
        num_to_check = row_to_check[self.column]
        return num_to_check >= self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.

        condition = ">="
        return "Is %s %s %s?" % (
            self.classes[self.column], condition, str(self.value))


def split_data(data_to_split, split_condition):
    true_data = []
    false_data = []
    for row in data_to_split:
        if split_condition.check(row):
            true_data.append(row)
        else:
            false_data.append(row)

    return true_data, false_data


def find_best_split(dataset):
    best_gain = 0
    best_question = None
    parent_entropy = find_entropy(dataset)
    n_features = len(dataset[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in dataset])  # unique values in column

        for val in values:
            split_condition = SplitCondition(col, val)

            # Try splitting dataset
            true_dataset, false_dataset = split_data(dataset, split_condition)

            # Skip this split if it doesn't divide
            if len(true_dataset) == 0 or len(false_dataset) == 0:
                continue

            # Calculate information gain from split
            entropy = (len(true_dataset) / len(dataset)) * find_entropy(true_dataset)
            entropy += (len(false_dataset) / len(dataset)) * find_entropy(false_dataset)
            gain = parent_entropy - entropy

            if gain >= best_gain:
                best_gain, best_question = gain, split_condition

    return best_gain, best_question


def find_entropy(dataset):
    labels_dict = find_distribution(dataset)
    labels = list(labels_dict)
    return -sum([labels_dict[letter] * math.log(labels_dict[letter]) for letter in labels])


def find_distribution(dataset):
    distribution = find_counts(dataset)

    for label in distribution.keys():
        total_num = distribution[label]
        distribution[label] = total_num / len(dataset)
    return distribution


def find_counts(dataset):
    counts = {}  # a dictionary of label -> count.
    for row in dataset:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

class Leaf:
    def __init__(self, dataset):
        self.predictions = find_counts(dataset)


class DecisionNode:
    def __init__(self, split_condition, true_branch, false_branch):
        self.split_condition = split_condition
        self.true_branch = true_branch
        self.false_branch = false_branch


def construct_tree(dataset):
    # Try partitioning the dataset on each attribute
    gain, split_condition = find_best_split(dataset)

    # Base case: no further information gain so return a leaf
    if gain == 0:
        return Leaf(dataset)

    # Found attribute to partition on
    true_dataset, false_dataset = split_data(dataset, split_condition)

    # Recursively build true branch
    true_branch = construct_tree(true_dataset)

    # Recursively build false branch
    false_branch = construct_tree(false_dataset)

    # Return question node
    return DecisionNode(split_condition, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    classes = ["A", "C", "E", "G", "O", "Q"]

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.split_condition))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
        self.decision_tree = None

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    

        dataset = np.concatenate((x, y), axis=1)

        self.decision_tree = construct_tree(dataset)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def return_decision_tree(self):
        return self.decision_tree

    def predict_helper(self, row, node):
        if isinstance(node, Leaf):
            print(node.predictions)
            a_list = list(node.predictions.keys())
            return a_list[0]

        if self.decision_tree.split_condition.check(row):
            return self.predict_helper(row, node.true_branch)
        else:
            return self.predict_helper(row, node.false_branch)

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################

        # remember to change this if you rename the variable
        for row in x:
            np.append(predictions, self.predict_helper(row, self.decision_tree))

        return predictions

    def prune(self, x_val, y_val):
        """ Post-prune your DecisionTreeClassifier given some optional validation dataset.

        You can ignore x_val and y_val if you do not need a validation dataset for pruning.

        Args:
        x_val (numpy.ndarray): Instances of validation dataset, numpy array of shape (L, K).
                           L is the number of validation instances
                           K is the number of attributes
        y_val (numpy.ndarray): Class labels for validation dataset, numpy array of shape (L, )
                           Each element in y is a str 
        """

        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        #######################################################################
        #                 ** TASK 4.1: COMPLETE THIS METHOD **
        #######################################################################

