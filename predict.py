import pickle

import keras
from keras.models import load_model

# load test dataset
testing_file = '../data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']

# prepare data for network
X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
X_test = X_test.astype('float32') / 255

num_classes = 43
y_test = keras.utils.to_categorical(y_test, num_classes)

# load traind model
model = load_model('./output/traffic_sings_model.h5')

# predict
y_pred = model.predict(X_test)

# show the true and the predicted classes for the test dataset
for i in range(len(y_test)):
    print("{:4d} : True={: <2}, Predicted={}".format(i, y_test[i].argmax(axis=-1), y_pred[i].argmax(axis=-1)))
