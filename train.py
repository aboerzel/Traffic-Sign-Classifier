# import the necessary packages
import csv
import pickle

import Augmentor
import keras
import numpy as np
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam

batch_size = 64

# optimizer = SGD(lr=0.01)
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam(lr=1e-4, clipnorm=0.001)


class LeNet:
    @staticmethod
    def build(num_classes):
        model = Sequential()

        # Layer 1
        # Conv Layer 1 => 28x28x6
        model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 3)))

        # Layer 2
        # Pooling Layer 1 => 14x14x6
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        # Conv Layer 2 => 10x10x16
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6)))

        # Layer 4
        # Pooling Layer 2 => 5x5x16
        model.add(MaxPooling2D(pool_size=2, strides=2))

        # Flatten
        model.add(Flatten())

        # Layer 5
        # Fully connected layer 1 => 120x1
        model.add(Dense(units=120, activation='relu'))

        model.add(Dropout(0.5))

        # Layer 6
        # Fully connected layer 2 => 84x1
        model.add(Dense(units=84, activation='relu'))

        model.add(Dropout(0.5))

        # Output Layer => num_classes x 1
        model.add(Dense(units=num_classes, activation='softmax'))

        model.summary()
        return model


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


# load classes and sign names from csv file
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

# prepare data for network
X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
x_train = X_train.astype('float32') / 255

X_valid = X_valid.reshape((X_valid.shape[0], 32, 32, 3))
X_valid = X_valid.astype('float32') / 255

X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
X_test = X_test.astype('float32') / 255

num_classes = len(sign_names)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# image augmentation
p = Augmentor.Pipeline()
p.skew(probability=0.5, magnitude=0.1)
p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)

datagen = p.keras_generator_from_array(X_train, y_train, batch_size=batch_size)

# build LeNet model
model = LeNet.build(num_classes)

# The function to optimize is the cross entropy between the true label and the output (softmax) of the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit_generator(datagen, validation_data=(X_valid, y_valid), steps_per_epoch=len(X_train) / batch_size, epochs=100)

# predict
y_pred = model.predict(X_test)
