import pickle

import keras
from keras.models import load_model

testing_file = '../data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']

X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
X_test = X_test.astype('float32') / 255

num_classes = 43
y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model('./output/traffic_sings_model.h5')

# predict
y_pred = model.predict(X_test)

# show the true and predicted class for the test dataset
for i in range(len(y_test)):
    print("{:4d} : True={}, Predicted={}".format(i, y_test[i], y_pred[i]))
