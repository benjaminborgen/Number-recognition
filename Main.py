import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import gzip
import pickle


with open('full_dataset.pkl', 'rb') as f:
    #data = pickle.load(f)

#print(data)
    train_set, valid_set, test_set = pickle.load(f)

train_x, train_y = train_set

plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()