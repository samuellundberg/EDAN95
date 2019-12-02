from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, Model
from keras import models
from keras import layers
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


#base_dir = '/Users/samuel/Documents/kurser/applied_ML/EDAN95/lab3/'
base_dir = '/Users/Sofie/PycharmProjects/EDAN95/lab3/'

train_dir = os.path.join(base_dir, 'lowers_split/train')
validation_dir = os.path.join(base_dir, 'lowers_split/validation')
test_dir = os.path.join(base_dir, 'lowers_split/test')
conv_base = InceptionV3(weights='imagenet', include_top=False)
for layer in conv_base.layers:
    layer.trainable = False

x = conv_base.output
y = GlobalAveragePooling2D()(x)
dense = Dense(256, activation='relu')(y)
predictions = Dense(5, activation='softmax')(dense)
model = Model(input=conv_base.input, output=predictions)
model.summary()
batch_size = 20
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])
model.compile(loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil(train_generator.samples/batch_size),
    epochs=30,
    validation_data=validation_generator,
    validation_steps=np.ceil(validation_generator.samples/batch_size))

model.save('flowers_small_pretrained_2.h5')

with open(base_dir + '/trainHistoryDict_pretrained_2.p', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

