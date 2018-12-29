import csv
import pickle

import keras
import numpy as np
from keras.models import load_model

# load test dataset
testing_file = '../data/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']


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

# prepare data for network
X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
X_test = X_test.astype('float32') / 255

num_classes = len(sign_names)
y_test = keras.utils.to_categorical(y_test, num_classes)

# load traind model
model = load_model('./output/traffic_sings_model.h5')

# predict
y_pred = model.predict(X_test)

# show the true and the predicted classes for a couple of items of the test dataset
min = 30
count = 20
for i, (y_t, y_p) in enumerate(zip(y_test[min:min+count], y_pred[min:min+count])):
    print("{:4d} : True={: <2}  Predicted={: <2}  {}"
          .format(i + min, y_t.argmax(axis=-1), y_p.argmax(axis=-1),
                  y_t.argmax(axis=-1) == y_p.argmax(axis=-1)))

# evaluate model
loss, acc = model.evaluate(X_test, y_test, batch_size=64)
print('Loss:     {:.2f}%'.format(loss * 100))
print('Accuracy: {:.2f}%'.format(acc * 100))
