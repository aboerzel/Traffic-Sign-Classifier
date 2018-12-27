import pickle
import numpy as np

import matplotlib.pyplot as plt
import random

import keras
import tensorflow as tf

# print(tf.__version__)
# print(keras.__version__)

training_file = '../data/train.p'
validation_file = '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Train: " + str(len(X_train)))

labels = np.unique(y_train)
train_grouped = np.array([list(X_train[y_train[0:] == i]) for i in labels])

# show image of 10 random data points
fig, axs = plt.subplots(len(labels), 5, figsize=(10, 200))
#fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()
for l in labels:
    group = train_grouped[l]
    for i in range(5):
        image = group[random.randint(0, len(group)-1)]
        index = l * 5 + i
        axs[index].axis('off')
        axs[index].imshow(image)
        axs[index].set_title(y_train[index])

plt.show()
