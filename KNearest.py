import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import Printer as pt

def main():

    data_mnist = datasets.load_digits()

    algorithm_name = "K-nearest neighbors"

    # 75% for training and  25% for testing
    (train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data_mnist.data),
                                                                      data_mnist.target, test_size=0.25, random_state=42)

    # Create the validation data
    (train_data, val_data, train_labels, val_labels) = train_test_split(train_data, train_labels,
                                                                    test_size=0.1, random_state=84)

    # show the sizes of each data split
    print("\nTraining data points: {}".format(len(train_labels)))
    print("Validation data points: {}".format(len(val_labels)))
    print("Testing data points: {}\n".format(len(test_labels)))

    # The range of K-neighbors
    k_values = range(1, 100, 1)
    accuracies = []

    pt.training(algorithm_name)

    # Go through the values of K's
    for k in range(1, 100, 1):
        # train the k-Nearest Neighbor classifier with the current value of K
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_data, train_labels)

        # Returns mean accuracy of data and label
        score = model.score(val_data, val_labels)
        print("K=%d | Accuracy=%.2f%%" % (k, score * 100))
        accuracies.append(score)

    i = int(np.argmax(accuracies))
    pt.printSpacing()
    print("The K that had the highest registered accuracy was:")
    print("K = %d of %.2f%%" % (k_values[i], accuracies[i] * 100))

    # Train the classifier using the best K
    model = KNeighborsClassifier(n_neighbors=k_values[i])
    model.fit(train_data, train_labels)
    # Predict the class label of given data
    predictions = model.predict(test_data)

    pt.cf_report(test_labels, predictions)
    pt.printSpacing()