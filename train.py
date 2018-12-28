# import the necessary packages
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


class LeNet:
    @staticmethod
    def build(classes):
        model = Sequential()

        # Layer 1
        # Conv Layer 1 => 28x28x6
        model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 3)))

        # Layer 2
        # Pooling Layer 1 => 14x14x6
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        # Conv Layer 2 => 10x10x16
        model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu'))

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

        # Output Layer => classes x 1
        model.add(Dense(units=classes, activation='softmax'))

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

X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
x_train = X_train.astype('float32')
x_train /= 255

classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, classes)

p = Augmentor.Pipeline()
p.skew(probability=0.5, magnitude=0.1)
p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)

datagen1 = p.keras_generator_from_array(X_train, y_train, batch_size=32)

# datagen = ImageDataGenerator(
# #     featurewise_center=True,
# #     featurewise_std_normalization=True,
# #     rotation_range=20,
# #     width_shift_range=0.2,
# #     height_shift_range=0.2,
# #     zoom_range=0.2,
# #     data_format='channels_last')

#datagen1.fit(X_train)

model = LeNet.build(classes)
opt = SGD(lr=0.01)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = Adam(lr=1e-4, clipnorm=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# The function to optimize is the cross entropy between the true label and the output (softmax) of the model
# We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
#model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=["accuracy"])

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=’SGD’, metrics=[“accuracy”])

model.fit_generator(datagen1, steps_per_epoch=len(X_train) / 32, epochs=100)

y_pred = model.predict(X_test)
