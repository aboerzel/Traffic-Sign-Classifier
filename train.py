import csv
import pickle

import Augmentor
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology as morp
from skimage.filters import rank
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import MaxPooling2D, Activation, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta

# hyperparameter for training
batch_size = 64
num_epochs = 100
optimizer_method = 'sdg'


# LeNet model architecture
class LeNet:
    @staticmethod
    def build(num_classes):
        model = Sequential()

        # Layer 1
        # Conv Layer 1 => 28x28x6
        model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 1)))

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

        # show and return the constructed network architecture
        model.summary()
        return model


class MiniVGGNet:
    @staticmethod
    def build(num_classes):
        model = Sequential()

        chanDim = -1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 1)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        # show and return the constructed network architecture
        model.summary()
        return model


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
print('Number of classes: {}'.format(num_classes))

# load train, validation and test dataset
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


# convert images to grayscale
def to_grayscale(images):
    result = []
    for image in images:
        result.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return np.array(result)


X_train = to_grayscale(X_train)
X_valid = to_grayscale(X_valid)
X_test = to_grayscale(X_test)


# apply local histogram equalization
def local_histogram_equalization(image):
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local


# X_train = np.array(list(map(local_histogram_equalization, X_train)))
# X_valid = np.array(list(map(local_histogram_equalization, X_valid)))
# X_test = np.array(list(map(local_histogram_equalization, X_test)))

# convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# image augmentation (https://augmentor.readthedocs.io/en/master/)
p = Augmentor.Pipeline()
p.zoom(probability=0.8, min_factor=0.8, max_factor=1.2)
p.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)
p.skew(probability=0.8, magnitude=0.2)

# adapt data to the network input
X_train = X_train.reshape((X_train.shape[0], 32, 32, 1))
X_valid = X_valid.reshape((X_valid.shape[0], 32, 32, 1))
X_test = X_test.reshape((X_test.shape[0], 32, 32, 1))

# normalize data from 0.0 to 1.0
# don't normalize X_train, because this is already done by batch normalization
# X_train = X_train.astype('float32') / 255
X_valid = X_valid.astype('float32') / 255
X_test = X_test.astype('float32') / 255

datagen = p.keras_generator_from_array(X_train, y_train, batch_size=batch_size)


def get_optimizer(optimizer_method):
    if optimizer_method == "sdg":
        return SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    if optimizer_method == "rmsprop":
        return RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adam":
        return Adam(lr=0.001, decay=0.001 / num_epochs)
        # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adagrad":
        return Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    if optimizer_method == "adadelta":
        return Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)


# https://medium.com/algoscale/how-to-plot-the-model-training-in-keras-using-custom-callback-function-and-using-tensorboard-41e4ce3cb401
class TrainingPlot(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = np.arange(0, len(self.losses))
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            # plt.style.use("seaborn")
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('output/Epoch-{}.png'.format(epoch))
            plt.close()


def get_callbacks(optimizer_method):
    model_filepath = './output/traffic_sings_model_{}.h5'.format(optimizer_method)
    callbacks = [
        TrainingPlot(),
        EarlyStopping(monitor='loss', min_delta=0, patience=5, mode='auto', verbose=1),
        ModelCheckpoint(model_filepath, save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=1e-4, cooldown=0,
                          min_lr=0)]
    return callbacks


def plot_train_history(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, len(H.history["acc"])), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, len(H.history["val_acc"])), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('./output/training-loss-and-accuracy_{}.png'.format(optimizer_method))
    plt.show()


# build model
model = MiniVGGNet.build(num_classes)

# the function to optimize is the cross entropy between the true label and the output (softmax) of the model
model.compile(optimizer=get_optimizer(optimizer_method), loss='categorical_crossentropy', metrics=['accuracy'])

# train model
H = model.fit_generator(datagen,
                        validation_data=(X_valid, y_valid),
                        steps_per_epoch=len(X_train) / batch_size,
                        callbacks=get_callbacks(optimizer_method),
                        epochs=num_epochs)

# plot and save the training loss and accuracy
plot_train_history(H)
