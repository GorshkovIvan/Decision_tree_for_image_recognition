import numpy as np
import unittest


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

    def distrbution_of_instances(self):
        instances_proportions = []
        rows, columns = np.shape(self.x)
        total_number_of_instances = rows
        for class_label in np.unique(self.y):
            class_instances = self.x[self.y == class_label]
            rows, column = np.shape(class_instances)
            instances_proportions.append(rows/total_number_of_instances)

        return instances_proportions


data_train_full = data_set("data/train_full.txt")

(x1, y1) = data_train_full.read_data()
proportions_full = data_train_full.distrbution_of_instances()
print(proportions_full)

data_train_sub = data_set("data/train_sub.txt")

(x2, y2) = data_train_sub.read_data()
proportions_sub = data_train_sub.distrbution_of_instances()
print(proportions_sub)


import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

print(np.unique(y1))
print(np.unique(y2))
df = pd.DataFrame(proportions_full, columns = ["Full set"])
df["Sub set"] = proportions_sub
df.index = np.unique(y1)
#sn.barplot(x = "Class", y= "Proportion", df['Full set'])
print(df)
dft = df.T
ax = sn.barplot(df["Full set"])
print(dft)
#sn.barplot(df['Sub set'])

#plt.show()

df.plot(kind="bar")
plt.title("Class distribution in two sets")
plt.xlabel("Classes")
plt.ylabel("Proportion")
plt.show()


