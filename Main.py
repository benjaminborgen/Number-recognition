import KNearest as kn
import DecisionTreeClassifier as dtc
import MLP as mlp
import NaiveBayes as nb
import Printer as pt

pt.displayHeader()
dtc.main()  # Decision tree classifier
nb.main()   # Naive bayes
kn.main()   # K-Nearest
mlp.main()  # Multilayer perceptron