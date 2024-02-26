# Modeling the probability of class membership using logistic classic_machine_learning (classifying algorithm)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
sc = StandardScaler()
x_train, x_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, stratify=iris.target, test_size=0.2, random_state=1)
# random_state - reproducibility of shuffling training data after each epoch

sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
reg = LogisticRegression(random_state=1)
reg.fit(x_train_std, y_train)
predicted = reg.predict(x_test_std)

print(f'Accuracy score: {accuracy_score(y_test, predicted)}')
markers = ['s', 'v', 'x']

x = iris.data
y = iris.target
for idx, cls in enumerate(np.unique(y_test)):
    plt.scatter(x_train[y_train == cls, 2], x_train[y_train == cls, 3], edgecolors='black', marker=markers[idx],
                label=cls)

plt.scatter(x_test[:, 2], x_test[:, 3], edgecolors='black', marker='o', c='white', label='Test set')
plt.scatter(x_test[y_test != predicted, 2], x_test[y_test != predicted, 3], edgecolors='black', marker='o', c='red',
            label='Wrong classified')

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend(loc='upper left')
plt.show()
