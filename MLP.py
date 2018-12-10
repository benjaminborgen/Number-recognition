import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
import os.path
import Printer as pt

def main():
    directory = 'mlp_model.pkl'
    mnist = fetch_mldata("MNIST original")

    pt.print_mlp()

    X, Y = mnist.data, mnist.target

    # Splits array or matrices into random train and test subsets.
    x_train, x_test, y_train, y_test = train_test_split(X / 255., Y, test_size=0.25)


    print("Fetched MNIST dataset with %d training and %d test samples" % (len(y_train), len(y_test)))
    print("Digit distribution in whole dataset:", np.bincount(Y.astype('int64')))

    clf = None
    if os.path.exists(directory):
        print("Loading the model from file...")
        clf = joblib.load(directory).best_estimator_
    else:
        print("Training the model...")
        # Sizes of the hidden layers
        params = {"hidden_layer_sizes": [(256,), (512,), (128, 256, 128,)]}
        mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
        clf.fit(x_train, y_train)
        print("Completed the grid search with best mean cross-validated score:", clf.best_score_)
        print("Best params appeared to be", clf.best_params_)
        joblib.dump(clf, directory)
        clf = clf.best_estimator_

    pt.printSpacing()
    score = clf.score(x_test, y_test)

    print("Accuracy: ", score)