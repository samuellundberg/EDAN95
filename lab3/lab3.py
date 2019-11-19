import pickle

from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np

import matplotlib.pyplot as plt

# **** Building a Simple Convolutional Neural Network ****
# 1.
# You will need to modify some parameters so that your network handles multiple classes.
# You will also adjust the number of steps so that your generator in the fitting procedure sees all the samples.
# You will report the training and validation losses and accuracies and comment on the possible overfit.

# Book: http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf

# Create model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5)) # added this layer to avoid overfittning
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))  # we have five classes, logistic function

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

base = '/Users/Sofie/PycharmProjects/EDAN95/lab3'
# base = '/Users/samuel/Documents/kurser/applied_ML/EDAN95/lab3'
train_dir = base + '/lowers_split/train'
validation_dir = base + '/lowers_split/validation'
test_dir = base + '/lowers_split/test'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])

# Fitting the network

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
model.summary()

#print(train_generator.samples)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(train_generator.samples / train_generator.batch_size),
    epochs=15,
    validation_data=validation_generator,
    validation_steps=np.ceil(validation_generator.samples / validation_generator.batch_size))

model.save('flowers_small_1_dropout_15_epocs.h5')

with open(base + '/trainHistoryDict_1_dropout_15_epocs.p', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
