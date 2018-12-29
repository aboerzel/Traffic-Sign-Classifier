import pickle
import numpy as np

import matplotlib.pyplot as plt
import random

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
n_rows = len(labels)
n_cols = 10
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 200))
fig.subplots_adjust(hspace=0.3)
axs = axs.ravel()
for row in range(0, n_rows):
    group = train_grouped[row]
    for col in range(n_cols):
        image = group[random.randint(0, len(group) - 1)]
        index = col + row * n_cols
        axs[index].axis('off')
        axs[index].imshow(image)
        axs[index].set_title(y_train[index])

plt.show()
