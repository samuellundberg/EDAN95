import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
import pickle

def extract_features(directory):

    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])
    features = np.zeros(shape=(generator.samples, 3, 3, 2048))  # 3,3,2048 is the outputsize from the convbase
    labels = np.zeros(shape=(generator.samples,5))
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if (i+1) * batch_size > generator.samples:
            break
    return features, labels


if __name__ == '__main__':
    #dubbelkolla hur man gör detta

    conv_base = InceptionV3(weights='imagenet', include_top=False)
    conv_base.summary()
    #base_dir = '/Users/Sofie/PycharmProjects/EDAN95/lab3/'

    base_dir = '/Users/samuel/Documents/kurser/applied_ML/EDAN95/lab3/'
    #
    train_dir = os.path.join(base_dir, 'lowers_split/train')
    validation_dir = os.path.join(base_dir, 'lowers_split/validation')
    test_dir = os.path.join(base_dir, 'lowers_split/test')

    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20
    train_features, train_labels = extract_features(train_dir)
    validation_features, validation_labels = extract_features(validation_dir)
    test_features, test_labels = extract_features(test_dir)

## The extracted features are currently of shape (samples, 4, 4, 512). You’ll feed them
## to a densely connected classifier, so first you must flatten them to (samples, 8192):

    train_features = np.reshape(train_features, (len(train_labels), 3*3*2048))
    validation_features = np.reshape(validation_features, (len(validation_labels), 3*3*2048))
    test_features = np.reshape(test_features, (len(test_labels), 3*3*2048))


    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=3*3*2048))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))
    model.save('flowers_small_pretrained.h5')

    with open(base_dir + '/trainHistoryDict_pretrained.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
