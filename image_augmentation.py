import pickle

import Augmentor
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2

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

classes = len(np.unique(y_train))


def to_grayscale(images):
    result = []
    for image in images:
        result.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(1, 32, 32))
    return np.array(result).astype(np.float32)


X_train = to_grayscale(X_train)
#X_train = X_train.reshape(X_train.shape[0], 1, 32, 32)
#X_train = X_train.astype('float32')

# X_test = X_test.reshape(X_test.shape[0], 1, 32, 32)


# X_test = X_test.astype('float32')

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2)

datagen.fit(X_train)

batch = datagen.flow(X_train, y_train, batch_size=32)
augmented_images, labels = next(batch)

# augmented_images = augmented_images.reshape(X_train.shape[0], 32, 32)
plt.imshow(augmented_images[0].reshape(32, 32), cmap='gray')
plt.show()

# p = Augmentor.Pipeline()
# p.greyscale(probability=1)
# p.skew(probability=0.5, magnitude=0.1)
# p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
# p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
#
# g = p.keras_generator_from_array(X_train, y_train, batch_size=32)
# augmented_images, labels = next(g)

cols = 1
n_images = len(augmented_images)

fig = plt.figure()

for n, image in enumerate(augmented_images):
    a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
    plt.axis("off")
    plt.imshow(image)
    a.set_title('Image (%d)' % (n + 1))

fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
plt.show()
