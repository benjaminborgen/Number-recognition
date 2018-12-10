from sklearn.metrics import classification_report

def displayHeader():
    print("Program running four different algorithms with the MNIST dataset.")
    print("Builds- and uses models to predict handwritten numbers in the MNIST dataset.")
    print("Reports the accuracy, precision, recall, F1-score and support.")
    printSpacing()

# Prints the result with a percentage.
def printResults(algorithm, count_of_hits):
    print("%s results:" % (algorithm))
    print("The algorithm predicted correct in %d of 21 000 cases." % (count_of_hits))
    print("Accuracy = ", (count_of_hits / 21000) * 100)
    printSpacing()

def printSpacing():
    print("------------------------------------")

def print_mlp():
    print("Fetching the MNIST-dataset")
    printSpacing()
    print("Running the multilayer perceptron algorithm...")
    printSpacing()

def training(algorithm_name):
    print("Training %s algorithm...\n" % (algorithm_name))

# Classification report print-out
def cf_report(test_labels, predictions):
    print("\nEvaluation of the test data:")
    print(classification_report(test_labels, predictions))
