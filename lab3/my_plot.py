import pickle

from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

base = '/Users/Sofie/PycharmProjects/EDAN95/lab3'
model = load_model('flowers_small_1_15_epocs.h5')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_dir = base + '/lowers_split/test'

test_generator = test_datagen.flow_from_directory(
    test_dir,
    shuffle=False,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'])

predictions = model.predict_generator(test_generator, steps=np.ceil(test_generator.samples / test_generator.batch_size))
labels = test_generator.classes
predictions = np.argmax(predictions,1)
print(predictions)
print(metrics.confusion_matrix(labels, predictions))
print(metrics.classification_report(labels, predictions))
print(metrics.accuracy_score(labels, predictions))


with open(base + '/trainHistoryDict_15_epocs.p', 'rb') as file_pi:
    history = pickle.load(file_pi)
##### try to plot, doesnt work
acc = history['acc']
val_acc = history['val_acc']
loss = history['loss']
val_loss = history['val_loss']
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
