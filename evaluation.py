import numpy as np
from numpy.random import default_rng


class Evaluate(object):
    def confusion_matrix(self, y_gold, y_prediction, class_labels=None):
        """ Compute the confusion matrix.

        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels
            class_labels (np.ndarray): a list of unique class labels.
                                   Defaults to the union of y_gold and y_prediction.

        Returns:
            np.array : shape (C, C), where C is the number of classes.
                       Rows are ground truth per class, columns are predictions
        """

        # if no class_labels are given, we obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if not class_labels:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        # for each correct class (row),
        # compute how many instances are predicted for each class (columns)
        for (i, label) in enumerate(class_labels):
            # get predictions where the ground truth is the current class label
            indices = (y_gold == label)
            gold = y_gold[indices]
            predictions = y_prediction[indices]

            # quick way to get the counts per label
            (unique_labels, counts) = np.unique(predictions, return_counts=True)

            # convert the counts to a dictionary
            frequency_dict = dict(zip(unique_labels, counts))

            # fill up the confusion matrix for the current row
            for (j, class_label) in enumerate(class_labels):
                confusion[i, j] = frequency_dict.get(class_label, 0)

        return confusion

    def accuracy_from_confusion(self, confusion):
        """ Compute the accuracy given the confusion matrix

        Args:
            confusion (np.ndarray): shape (C, C), where C is the number of classes.
                        Rows are ground truth per class, columns are predictions

        Returns:
            float : the accuracy
        """

        if np.sum(confusion) > 0:
            return np.sum(np.diag(confusion)) / np.sum(confusion)
        else:
            return 0.

    def precision(self, confusion):
        """ Compute the precision score per class given the ground truth and predictions

        Also return the macro-averaged precision across classes.

        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels

        Returns:
            tuple: returns a tuple (precisions, macro_precision) where
                - precisions is a np.ndarray of shape (C,), where each element is the
                  precision for class c
                - macro-precision is macro-averaged precision (a float)
        """
        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])

                # Compute the macro-averaged precision
        macro_p = 0.
        if len(p) > 0:
            macro_p = np.mean(p)

        return (p, macro_p)

    def recall(self, confusion):
        """ Compute the recall score per class given the ground truth and predictions

        Also return the macro-averaged recall across classes.

        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels

        Returns:
            tuple: returns a tuple (recalls, macro_recall) where
                - recalls is a np.ndarray of shape (C,), where each element is the
                    recall for class c
                - macro-recall is macro-averaged recall (a float)
        """

        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])

                # Compute the macro-averaged recall
        macro_r = 0.
        if len(r) > 0:
            macro_r = np.mean(r)

        return (r, macro_r)

    def f1_score(self, confusion):
        """ Compute the F1-score per class given the ground truth and predictions

        Also return the macro-averaged F1-score across classes.

        Args:
            y_gold (np.ndarray): the correct ground truth/gold standard labels
            y_prediction (np.ndarray): the predicted labels

        Returns:
            tuple: returns a tuple (f1s, macro_f1) where
                - f1s is a np.ndarray of shape (C,), where each element is the
                  f1-score for class c
                - macro-f1 is macro-averaged f1-score (a float)
        """

        (precisions, macro_p) = self.precision(confusion)
        (recalls, macro_r) = self.recall(confusion)

        # just to make sure they are of the same length
        assert len(precisions) == len(recalls)

        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1
        macro_f = 0.
        if len(f) > 0:
            macro_f = np.mean(f)

        return (f, macro_f)

    def k_fold_split(self, n_splits, n_instances, random_generator=default_rng()):
        """ Split n_instances into n mutually exclusive splits at random.

        Args:
            n_splits (int): Number of splits
            n_instances (int): Number of instances to split
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list (length n_splits). Each element in the list should contain a
                numpy array giving the indices of the instances in that split.
        """

        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = random_generator.permutation(n_instances)

        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, n_splits)

        return split_indices

    def train_test_k_fold(self, n_folds, n_instances):
        """ Generate train and test indices at each fold.

        Args:
            n_folds (int): Number of folds
            n_instances (int): Total number of instances
            random_generator (np.random.Generator): A random generator

        Returns:
            list: a list of length n_folds. Each element in the list is a list (or tuple)
                with two elements: a numpy array containing the train indices, and another
                numpy array containing the test indices.
        """

        # split the dataset into k splits
        split_indices = self.k_fold_split(n_folds, n_instances)

        folds = []
        for k in range(n_folds):
            # pick k as test
            test_indices = split_indices[k]

            # combine remaining splits as train
            # this solution is fancy and worked for me
            # feel free to use a more verbose solution that's more readable
            train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

            folds.append([train_indices, test_indices])

        return folds

    def average_accuracy_across_k_folds(self, n_folds, x_full, y_full, classifier):
        accuracies = np.zeros((n_folds,))
        max_accuracy = 0
        trained_trees_list = []

        for i, (train_indices, test_indices) in enumerate(
                self.train_test_k_fold(n_folds, len(x_full))):
            # get the dataset from the correct splits
            x_train = x_full[train_indices, :]
            y_train = y_full[train_indices]
            x_test = x_full[test_indices, :]
            y_test = y_full[test_indices]

            trained_trees_list.append(classifier.fit(x_train, y_train))
            y_predicted = classifier.predict(x_test)

            confusion_matrix = self.confusion_matrix(y_test, y_predicted)
            accuracy = self.accuracy_from_confusion(confusion_matrix)

            if accuracy > max_accuracy:
                max_accuracy = accuracy

            accuracies[i] = accuracy

        print("Accuracies for %s folds:" % n_folds)
        print(accuracies)
        print("Mean accuracy for %s folds:" % n_folds)
        print(accuracies.mean())
        print("Standard deviation for %s folds:" % n_folds)
        print(accuracies.std())
        print("Max accuracy for %s folds:" % n_folds)
        print(max_accuracy)

        # return the decision tree model with highest accuracy
        trained_trees = np.array(trained_trees_list)
        return trained_trees, trained_trees[np.argmax(accuracies)]
