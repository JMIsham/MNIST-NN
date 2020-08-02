import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.constraints import max_norm
from keras.utils.vis_utils import plot_model
plt.rcParams['figure.figsize'] = (7,7)

def print_images(X_train, y_train):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(X_train[i].reshape(28,28), interpolation='none')
        plt.title("Class {}".format(y_train[i]))

def prepare_data(X_train,X_test,y_train,y_test):
    # reshape the data
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalize the Pixel values
    X_train /= 255
    X_test /= 255    
    
    # Converting to catagorical variable 
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    # printing the loaded images
    print_images(X_train, y_train)
    
    return X_train, X_test, Y_train, Y_test, y_test

def load_fresh_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return prepare_data(X_train,X_test,y_train,y_test)

X_train, X_test, Y_train, Y_test, y_test = load_fresh_data()

def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, epochs=4):
    model.fit(X_train, Y_train,
          batch_size=128, epochs=epochs,
        verbose=1,
          validation_data=(X_test, Y_test))
    return model

def predict_using_model(model, X_test, y_test):
    predicted_classes = model.predict_classes(X_test)
    
    # get the correct and incorrect predictions seperated so that it can be printed later
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    
    return predicted_classes, correct_indices, incorrect_indices


def plot_correct_and_incorrect(predicted_classes, correct_indices, incorrect_indices, y_test ):
    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(X_test[correct].reshape(28,28), interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(X_test[incorrect].reshape(28,28), interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

model = get_model()
model = train_model(model, X_train, Y_train)
predicted_classes, correct_indices, incorrect_indices = predict_using_model(model, X_test, y_test)
plot_correct_and_incorrect(predicted_classes, correct_indices, incorrect_indices, y_test )
print ("Test Accuracy : " + str(len(correct_indices)/(len(predicted_classes))))
model.summary()