from classification import *
from read_data import read_dataset
import numpy as np

(x_full, y_full, classes_full) = read_dataset("train_full.txt")
#dataset = ((x_full, y_full))
y_full = np.reshape(y_full, (-1, 1))
(x_test, y_test, classes_test) = read_dataset("test.txt")

my_classifier = DecisionTreeClassifier()
my_classifier.fit(x_full, y_full)
print_tree(my_classifier.decision_tree)
print(my_classifier.predict(x_test))
print(y_test)

#tree = construct_tree(dataset)
#print_tree(tree)


