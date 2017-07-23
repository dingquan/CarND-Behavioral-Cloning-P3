import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def resize_and_normalization(input):
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(input, (64, 64))
    resized = resized / 255.0 - 0.5
    return resized

lines = []
with open('../p3/both-tracks/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    if (line[0] == 'center'): # skip header line
        continue
    source_path_center = line[0]
    filename_center = source_path_center.split('/')[-1]
    current_path_center = '../p3/both-tracks/IMG/' + filename_center
    image_center = cv2.imread(current_path_center)
    image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
    images.append(image_center)
    measurement = float(line[3])
    measurements.append(measurement)
    # flip the image
    image_flipped = np.fliplr(image_center)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)
    # use the side camera images
    # correction = 0.1

    # source_path_left = line[1]
    # filename_left = source_path_left.split('/')[-1]
    # current_path_left = '../p3/data/IMG/' + filename_left
    # image_left = cv2.imread(current_path_left)
    # image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    # images.append(image_left)
    # measurement_left = measurement + correction
    # measurements.append(measurement_left)
    # image_flipped_left = np.fliplr(image_left)
    # measurement_flipped_left = -measurement_left
    # images.append(image_flipped_left)
    # measurements.append(measurement_flipped_left)

    # source_path_right = line[2]
    # filename_right = source_path_right.split('/')[-1]
    # current_path_right = '../p3/data/IMG/' + filename_right
    # image_right = cv2.imread(current_path_right)
    # image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    # images.append(image_right)
    # measurement_right = measurement + correction
    # measurements.append(measurement_right)
    # image_flipped_right = np.fliplr(image_right)
    # measurement_flipped_right = -measurement_right
    # images.append(image_flipped_right)
    # measurements.append(measurement_flipped_right)

X_train = np.array(images)
y_train = np.array(measurements)

X_train, y_train = shuffle(X_train, y_train, random_state=12345)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: my_resize_function(x)))
# model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Lambda(resize_and_normalization))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')

