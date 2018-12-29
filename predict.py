import csv
import os

import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

filename = "image2.jpg"


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

# read image file an preprocess image
image = cv2.imread(os.path.join('test_images', filename))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)

X_test = np.array([resized_image])
X_test = X_test.astype('float32') / 255

# load traind model
model = load_model('./output/traffic_sings_model.h5')

# predict
y_pred = model.predict(X_test)
class_name = sign_names[y_pred.argmax(axis=-1)[0]][1]

print("{} => {}".format(filename, class_name))

plt.imshow(image)
plt.title(class_name)
plt.show()

