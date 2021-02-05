import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class data_set:

    def __init__(self, path):
        self.filepath = path
        self.x, self.y = self.read_data()

    def print_data(self):
        for line in open(self.filepath):
            print(line.strip())

    def read_data(self):

        x = []
        y_labels = []

        for line in open(self.filepath):
            string_line = line.strip().split(",")
            if not line.strip() == '':
                y_labels.append(string_line[-1])
                x.append(list(map(float, string_line[:-1])))

        x = np.array(x)
        y_labels = np.array(y_labels)

        return x, y_labels

    def print_dataset_summary_statistics(self):

        classes = np.unique(y)

        print("Minimum Values by Feature:")
        print(str(x.min(axis=0)))
        print('\n')
        print("Mean Values by Feature:")
        print(str(x.mean(axis=0)))
        print('\n')
        print("Median Values by Feature:")
        print(str(np.median(x, axis=0)))
        print('\n')
        print("Standard Deviation by Feature:")
        print(str(x.std(axis=0)))
        print('\n')
        print("Variance by Feature:")
        print(str(x.var(axis=0)))
        print('\n')

    def print_summary_statistics_by_class(self):

        classes = np.unique(y)

        for class_label in np.unique(y):
            print("Class: " + class_label)
            print('\n')
            x_class = self.x[self.y == class_label]
            print("Minimum Values by Feature:")
            print(str(x_class.min(axis=0)))
            print('\n')
            print("Mean Values by Feature:")
            print(str(x_class.mean(axis=0)))
            print('\n')
            print("Median Values by Feature:")
            print(str(np.median(x_class, axis=0)))
            print('\n')
            print("Standard Deviation by Feature:")
            print(str(x_class.std(axis=0)))
            print('\n')
            print("Variance by Feature:")
            print(str(x_class.var(axis=0)))
            print('\n')

    def distribution_of_instances(self):
        instances_proportions = []
        rows, columns = np.shape(self.x)
        total_number_of_instances = rows
        for class_label in np.unique(self.y):
            class_instances = self.x[self.y == class_label]
            rows, column = np.shape(class_instances)
            instances_proportions.append(rows/total_number_of_instances)

        return instances_proportions

    def absolute_distribution_of_instances(self):

        absolute_instances_proportions = []

        for class_label in np.unique(self.y):
            class_instances = self.x[self.y == class_label]
            rows, column = np.shape(class_instances)
            absolute_instances_proportions.append(rows)

        return absolute_instances_proportions


data_train_full = data_set("data/train_full.txt")

(x1, y1) = data_train_full.read_data()
proportions_full = data_train_full.distrbution_of_instances()
print(proportions_full)

data_train_sub = data_set("data/train_sub.txt")

(x2, y2) = data_train_sub.read_data()
proportions_sub = data_train_sub.distrbution_of_instances()
print(proportions_sub)


print(np.unique(y1))
print(np.unique(y2))
df = pd.DataFrame(proportions_full, columns = ["Full set"])
df["Sub set"] = proportions_sub
df.index = np.unique(y1)
print(df)

"""
df.plot(kind="bar")
plt.title("Class distribution in two sets")
plt.xlabel("Classes")
plt.ylabel("Proportion")
plt.show()
"""
# Question 1.3

data_noisy = data_set("data/train_noisy.txt")

x_noisy, y_noisy = data_noisy.read_data()
#y_differences = y1[y_noisy != y1 and x_noisy == x1]
#print(len(y_differences))
print(len(y_noisy))
print(len(y1))
print(np.array_equal(x_noisy, x1))

xy_noisy = np.column_stack((x_noisy, y_noisy))
xy1 = np.column_stack((x1, y1))


sorted_noisy = np.sort(xy_noisy, axis=0)
sorted_full = np.sort(xy1, axis=0)
print(np.array_equal(sorted_noisy[:-2], sorted_full[:-2]))
#print(sorted_noisy[:-2])

#print(sorted_noisy[456])
#print(sorted_full[456])
differences_sorted = sorted_full[sorted_full != sorted_noisy]
print(differences_sorted)

proportions_noisy = data_noisy.absolute_distrbution_of_instances()

proportions_full = data_train_full.absolute_distrbution_of_instances()
df_noisy = pd.DataFrame(proportions_full, columns = ["Full set"])
df_noisy["Noisy set"] = proportions_noisy

df_noisy.index = np.unique(y1)
print(df_noisy)

df_noisy.plot(kind="bar")
plt.title("Class distribution in two sets")
plt.xlabel("Classes")
plt.ylabel("Proportion")
plt.show()



