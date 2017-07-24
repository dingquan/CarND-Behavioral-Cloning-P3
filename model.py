import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def resize_and_normalization(input):
    from keras.backend import tf as ktf
    resized = ktf.image.resize_images(input, (64, 64))
    resized = resized / 255.0 - 0.5
    return resized

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                filename_center = batch_sample[0].split('/')[-1]
                path_center = '../p3/both-tracks/IMG/' + filename_center
                image_center = cv2.imread(path_center)
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
                images.append(image_center)
                angle_center = float(batch_sample[3])
                angles.append(angle_center)

                # flip the image
                image_flipped = np.fliplr(image_center)
                angle_flipped = -angle_center
                images.append(image_flipped)
                angles.append(angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Read lines from CSV file
samples = []
with open('../p3/both-tracks/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Shuffle the samples before splitting to test and validation set
# because in driving_log.csv file, data for track 2 follows data for track 1
# we want to shuffle it so validation set don't end up with data from track 2 only
shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# This is the model published by NVIDIA (Except for the resizing)
model = Sequential()
model.add(Cropping2D(cropping=((20,20), (0,0)), input_shape=(160,320,3)))
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
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
# We are flipping images in the generator, so we're creating 2x number samples as input
model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples) * 2, 
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples) * 2,
    nb_epoch=5)

model.save('model.h5')
