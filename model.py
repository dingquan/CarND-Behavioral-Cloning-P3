import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D

def my_resize_function(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (45, 160))

lines = []
with open('../p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    if (line[0] == 'center'): # skip header line
        continue
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../p3/data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    # flip the image
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: my_resize_function(x)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

model.save('model.h5')

