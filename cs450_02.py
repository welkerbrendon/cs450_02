import difflib

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from KNNClassifier import KNNClassifier

iris = datasets.load_iris()

from sklearn.model_selection import train_test_split

csv = False
response = input("Would you like to use your own CSV file? (y/n): ")
if(response.__contains__("n")):
    iris = datasets.load_iris()
else:
    file_name = input("Please enter the path to the CSV file: ")
    import csv
    f = open(file_name)
    reader = csv.reader(f)
    item = []
    list_of_list = []
    for row in reader:
        for col in row:
            item.append(col)
        list_of_list.append(row)
    reader.close()
    csv = True

import numpy as np
from sklearn.model_selection import train_test_split
if(csv):
    data = []
    target = []
    for list in iris:
        data.append(list[0, len(iris) - 2])
        target.append(list[len(iris) - 1])
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.3)
else:
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=.3)

classifier = KNNClassifier(n_neighbors=int(input("Please enter the K value you would like to run with: ")))
model = classifier.fit(data_train, target_train)

possible_targets = []
keep_going = True
while(keep_going):
    input_target = int(input("Please enter all possible targets (numbers only), enter -1 when finished: "))
    if(input_target != -1):
        possible_targets.append(input)
    else:
        keep_going = False

targets_predicted = model.predict(data_test, possible_targets)

classifier2 = KNeighborsClassifier(n_neighbors=3)
model2 = classifier.fit(data_train, target_train)
targets_predicted2 = model.predict(data_test)

similarity_amount = difflib.SequenceMatcher(None, targets_predicted, target_test)
similarity_amount2 = difflib.SequenceMatcher(None, targets_predicted2, target_test)
print("My prediction: \n", similarity_amount.ratio())
print("Built-in KNeightborsClassier prediction: \n", similarity_amount2.ratio())