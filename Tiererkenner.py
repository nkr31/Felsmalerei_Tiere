# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array,load_img


#def load_pictures():
   # train_datagen = ImageDataGenerator(
       # rescale=1. / 255,
       # shear_range=0.2,
       # zoom_range=0.2,
       # horizontal_flip=True)

  #  print("train_datagen erzeugt")

   # test_datagen = ImageDataGenerator(rescale=1. / 255)
 #   print("test_datagen erzeugt")

   # train_generator = train_datagen.flow_from_directory(
   #     'data/train',
    #    target_size=(150, 150),
    #    batch_size=32,
    #    class_mode='categorical')
  #  print("train_generator erzeugt")

  #  validation_generator = test_datagen.flow_from_directory(
    #    'data/test',
     #   target_size=(150, 150),
     #   batch_size=32,
     #   class_mode='binary')
  #  print("validation_generator erzeugt")

def create_model():
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3), data_format='channels_last'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print("model erstellt und compiliert")
    return model

def fit_model(model):
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=8,
        validation_data=validation_generator,
        validation_steps=100)
    print("training abgeschlossen")
    return model

def evaluate_model(model):
    score = model.evaluate_generator(validation_generator)
    print(score)

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    return model


train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

print("train_datagen erzeugt")

test_datagen = ImageDataGenerator(rescale=1. / 255)
print("test_datagen erzeugt")

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical')
print("train_generator erzeugt")

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical')
print("validation_generator erzeugt")
model=create_model()
model= fit_model(model)
evaluate_model(model)
save_model(model)


