# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json


def load_data():
     # 4. Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (
    X_test, y_test) = mnist.load_data()  # brauchen wir nicht, stattdessen unseren Bildordner aufrufen

    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)  # die null hat was mit der Anzahl zu tun, also die 60000 Bilder. 1: nur ein Farbkanal, d.h. schwarzweiß, dann PixelxPixel, für uns 200x200
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # teilt die Farbwerte durch 255, d.h. Farbwerte sind von 0 bis 1, daher auch Float
    X_test /= 255

    # 6. Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)  # stattdessen iwie Tiernamen übergeben, an der Stelle andere Tutorials vergleichen
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_test, X_train, Y_test, Y_train


def define_model_architecture():
    # 7. Define model architecture
    model = Sequential()  # können wir das Model ähnlich machen?

    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28), data_format='channels_first'))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def fit_model(model):
    model.fit(X_train, Y_train,
              batch_size=32, epochs=10, verbose=1)
    return model

def evaluate_model(model):
    score = model.evaluate(X_test, Y_test, verbose=0)
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

(X_train, y_train), ( X_test, y_test) = mnist.load_data()  # brauchen wir nicht, stattdessen unseren Bildordner aufrufen

 # 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # teilt die Farbwerte durch 255, d.h. Farbwerte sind von 0 bis 1, daher auch Float
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = load_model()
compile_model(model)
evaluate_model(model)