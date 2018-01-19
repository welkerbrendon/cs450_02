import difflib

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=.3)

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

similarity_amount = difflib.SequenceMatcher(None, targets_predicted, target_test)
print(similarity_amount.ratio())