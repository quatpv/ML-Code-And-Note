from __future__ import division
from sklearn import datasets

from utils.data_manipulation import train_test_split, normalize
from utils.data_operation import accuracy_score

from src.supervised_learning.logistic_regression.logistic_regression import LogisticRegression

def main():
    data = datasets.load_iris()

    # Using only two flower for this example
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=123)

    model = LogisticRegression(gradient_descent=False)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()