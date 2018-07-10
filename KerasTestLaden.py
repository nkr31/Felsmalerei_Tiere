# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist  # brauchen wir nicht!!
from keras.models import model_from_json

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # brauchen wir nicht, stattdessen unseren Bildordner aufrufen

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28) #die null hat was mit der Anzahl zu tun, also die 60000 Bilder. 1: nur ein Farbkanal, d.h. schwarzweiß, dann PixelxPixel, für uns 200x200
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 #teilt die Farbwerte durch 255, d.h. Farbwerte sind von 0 bis 1, daher auch Float
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)  # stattdessen iwie Tiernamen übergeben, an der Stelle andere Tutorials vergleichen
Y_test = np_utils.to_categorical(y_test, 10)

# 7. Define model architecture

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)