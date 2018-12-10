import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import Printer as pt


def main():
    data = pd.read_csv("train.csv").values
    algorithm_name = "Decision tree classifier"

    clf = DecisionTreeClassifier()

    # Training data for the algorithm
    x_train = data[0:21000, 1:]
    # Training label is the correct number of the images
    train_label = data[0:21000, 0]

    # Build a decision tree classifier from the training data and training label
    clf.fit(x_train, train_label)

    pt.training(algorithm_name)

    # Test data
    x_test = data[21000:, 1:]
    actual_label = data[21000:, 0]

    d = x_test[2]
    d.shape = (28, 28)

    # Returns the predicted for each sample
    predictions = clf.predict(x_test)

    count = 0
    for i in range(0,21000):
        count += 1 if predictions[i]==actual_label[i] else 0

    pt.printResults(algorithm_name, count)
    pt.cf_report(actual_label, predictions)
    pt.printSpacing()