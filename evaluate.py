import csv
import pickle

import cv2
import keras
import numpy as np
import skimage.morphology as morp
from skimage.filters import rank
from keras.models import load_model

# hyperparameter for evaluation
optimizer_method = 'rmsprop'


# load class-ids and sign names from csv file
def load_signnames_from_csv(filename):
    rows = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            class_id = row[0]
            sign_name = row[1]
            rows.append((class_id, sign_name))

    return np.array(rows)


sign_names = load_signnames_from_csv('signnames.csv')
num_classes = len(sign_names)

# load test dataset
testing_file = '../data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']


# convert images to grayscale
def to_grayscale(images):
    result = []
    for image in images:
        result.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return np.array(result)


#X_test = to_grayscale(X_test)


# apply local histogram equalization
def local_histogram_equalization(image):
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


#X_test = np.array(list(map(local_histogram_equalization, X_test)))

# convert class vector to binary class matrix.
y_test = keras.utils.to_categorical(y_test, num_classes)

# reshape data for training with LeNet
#X_test = X_test.reshape((X_test.shape[0], 32, 32, 1))

# normalize data between 0.0 and 1.0
X_test = X_test.astype('float32') / 255

# load trained model
model = load_model('./output/traffic_signs_model_{}.h5'.format(optimizer_method))

# print loss and accuracy of the trained model
loss, acc = model.evaluate(X_test, y_test, batch_size=64)
print('Loss:     {:.2f}%'.format(loss * 100))
print('Accuracy: {:.2f}%'.format(acc * 100))

# show the true and the predicted classes for a couple of items of the test dataset
y_pred = model.predict(X_test)

print()

start = 110
count = 20
for i, (y_t, y_p) in enumerate(zip(y_test[start:start + count], y_pred[start:start + count])):
    print("{:4d} : True={: <2}  Predicted={: <2}  {}"
          .format(i + start, y_t.argmax(axis=-1), y_p.argmax(axis=-1),
                  y_t.argmax(axis=-1) == y_p.argmax(axis=-1)))
