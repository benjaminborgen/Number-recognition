import pandas as pd
from sklearn.naive_bayes import GaussianNB
import Printer as pt

def main():
    dataset = pd.read_csv("train.csv").values
    algorithm_name = "Naive Bayes"

    clf = GaussianNB()

    # Training data for the algorithm
    # Training label is the correct number of the images
    x_train = dataset[0:21000, 1:]
    train_label = dataset[0:21000, 0]

    # Fit Gaussian Naive Bayes according to X, y
    clf.fit(x_train, train_label)

    pt.training(algorithm_name)

    x_test = dataset[21000:, 1:]
    actual_label = dataset[21000:, 0]

    d = x_test[3]
    d.shape = (28, 28)

    # Performs classification on array of vectors
    p = clf.predict(x_test)

    count = 0
    for i in range(0, 21000):
        count += 1 if p[i] == actual_label[i] else 0

    pt.printResults(algorithm_name, count)

    pt.cf_report(actual_label, p)
    pt.printSpacing()
