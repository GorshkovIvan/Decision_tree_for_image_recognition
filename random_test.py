from classification import *
from evaluation import Evaluate
from read_data import read_dataset

import numpy as np

(x_full, y_full, classes_full) = read_dataset("train_full.txt")

(x_test, y_test, classes_test) = read_dataset("test.txt")

(x_val, y_val, classes_val) = read_dataset("validation.txt")

my_classifier = DecisionTreeClassifier()
my_classifier.fit(x_full, y_full)
#print_tree(my_classifier.decision_tree)

y_predicted = my_classifier.predict(x_test)


#y_matches = y_predicted[y_predicted == y_test]
#print(y_matches.shape)

count = 0
for i in y_predicted:
    if y_predicted[i] == y_test[i]:
        count = count + 1
print("Percent correct:")
print((count/200) * 100)

evaluator = Evaluate()
confusion_matrix = evaluator.confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:")
print(confusion_matrix)
print("Accuracy:")
print(evaluator.accuracy_from_confusion(confusion_matrix))
print("Precision:")
print(evaluator.precision(confusion_matrix))
print("Recall:")
print(evaluator.recall(confusion_matrix))
print("F1 score:")
print(evaluator.f1_score(confusion_matrix))

"""
n_folds = 10
accuracies = np.zeros((n_folds, ))

array_tree, best_tree = evaluator.average_accuracy_across_k_folds(10, x_full, y_full, my_classifier)

# predict test data with the best tree model from the k-folds
my_classifier.decision_tree = best_tree
y_predicted = my_classifier.predict(x_test)

count = 0
for i in y_predicted:
    if y_predicted[i] == y_test[i]:
        count = count + 1
print("Percent correct:")
print((count/200) * 100)

row, col = x_test.shape
tree_predictions = np.empty((10, row))
np_tree = np.array(tree_predictions)

for i, tree in enumerate(array_tree):
    my_classifier.decision_tree = tree
    y_predicted = my_classifier.predict(x_test)
    reshape_y = [y_predicted]
    y_predicted = np.array(reshape_y)
    tree_predictions[i] = y_predicted


#y_counts = np.zeroes((6,))
arg_predictions = []
for i, row in enumerate(tree_predictions.T):
    unique, counts = np.unique(row, return_counts=True)
    arg_predictions.append(unique[np.argmax(counts)])

print(arg_predictions)
"""

print("Accuracy:")
print(evaluator.accuracy_from_confusion(confusion_matrix))

my_classifier.prune(x_val, y_val)

y_predict_val = my_classifier.predict(x_test)
confusion_matrix = evaluator.confusion_matrix(y_test, y_predict_val)
print("Accuracy pruned:")
print(evaluator.accuracy_from_confusion(confusion_matrix))
print("Precision pruned:")
print(evaluator.precision(confusion_matrix))
print("Recall pruned:")
print(evaluator.recall(confusion_matrix))
print("F1 score pruned:")
print(evaluator.f1_score(confusion_matrix))




