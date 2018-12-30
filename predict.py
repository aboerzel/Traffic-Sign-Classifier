import csv
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

# hyperparameter for evaluation
optimizer_method = 'sdg'


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

# read and preprocess test images
original_images = []
X_test = []
filenames = glob.glob('./test_images/*.jpg')
for filename in filenames:
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_images.append(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).reshape(32, 32, 1)
    X_test.append(resized_image)

X_test = np.array(X_test)

# normalize data from 0.0 to 1.0
X_test = X_test.astype('float32') / 255

# load trained model
model = load_model('./output/traffic_sings_model_{}.h5'.format(optimizer_method))

# predict
y_pred = model.predict(X_test)

# print class predictions
num_files = int(len(filenames))
cols = 5
rows = int(num_files / cols)
if num_files % cols > 0:
    rows += 1

fig, axs = plt.subplots(rows, cols, figsize=(10, 200))
axs = axs.ravel()

for i, (filename, image, org_image) in enumerate(zip(filenames, X_test, original_images)):
    class_name = sign_names[y_pred.argmax(axis=-1)[i]][1]
    print("{} => {}".format(filename, class_name))
    axs[i].axis('off')
    axs[i].imshow(org_image)
    axs[i].set_title(class_name)

plt.show()
