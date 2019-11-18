from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
# **** Building a Simple Convolutional Neural Network ****
# 1.
# You will need to modify some parameters so that your network handles multiple classes.
# You will also adjust the number of steps so that your generator in the fitting procedure sees all the samples.
#
# You will report the training and validation losses and accuracies and comment on the possible overfit.
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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_dir = '/Users/Sofie/PycharmProjects/EDAN95/lab3/lowers_split/train'
validation_dir = '/Users/Sofie/PycharmProjects/EDAN95/lab3/lowers_split/validation'

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
