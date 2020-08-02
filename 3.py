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
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255    
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    print_images(X_train, y_train)
    return X_train, X_test, Y_train, Y_test, y_test

def load_fresh_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return prepare_data(X_train,X_test,y_train,y_test)

def get_noisy_data_2(noise_factor  = 0.25):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test, Y_train, Y_test, y_test = prepare_data(X_train,X_test,y_train,y_test)
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)
    print_images(X_train_noisy, y_train)
    return X_train_noisy, X_test_noisy, Y_train, Y_test, y_test

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

X_train, X_test, Y_train, Y_test, y_test = load_fresh_data()
X_train_noisy, X_test_noisy, Y_train, Y_test, y_test = get_noisy_data_2(noise_factor  = 0.50)

def train_model(model, X_train, Y_train, epochs=4):
    model.fit(X_train, Y_train,
          batch_size=128, epochs=epochs,
         verbose=1,
          validation_data=(X_test, Y_test))
    return model

def predict_using_model(model, X_test, y_test):
    predicted_classes = model.predict_classes(X_test)
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

# auto encorder model explained in the report
max_norm_value = 2.0

auto_encoder = Sequential()
auto_encoder.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=(28,28,1)))
auto_encoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2DTranspose(32, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2DTranspose(64, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
auto_encoder.add(Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
auto_encoder.fit(X_train_noisy, X_train,
                epochs=4,
                batch_size=150,
                validation_split=0.2)

# only denoise 4 images for the sake of printing 
number_of_visualizations = 4
samples = X_test_noisy[:number_of_visualizations]
targets = X_test[:number_of_visualizations]
denoised_images = auto_encoder.predict(samples)

# print the denoised images
for i in range(0, number_of_visualizations):
  noisy_image = X_test_noisy[i][:, :, 0]
  pure_image  = X_test[i][:, :, 0]
  denoised_image = denoised_images[i][:, :, 0]
  input_class = y_test[i]
  fig, axes = plt.subplots(1, 3)
  fig.set_size_inches(8, 3.5)
  axes[0].imshow(noisy_image)
  axes[0].set_title('Noisy image')
  axes[1].imshow(pure_image)
  axes[1].set_title('Pure image')
  axes[2].imshow(denoised_image)
  axes[2].set_title('Denoised image')
  fig.suptitle(f'MNIST target = {input_class}')
  plt.show()

# de noising pre processing
X_train_denoised =  auto_encoder.predict(X_train_noisy)
X_test_denoised =  auto_encoder.predict(X_test_noisy)

# user the same model but in this case we are using denising as a pre processing 
model = get_model()
model = train_model(model, X_train_denoised, Y_train)
predicted_classes, correct_indices, incorrect_indices = predict_using_model(model, X_test_denoised, y_test)
plot_correct_and_incorrect(predicted_classes, correct_indices, incorrect_indices, y_test )
print ("Test Accuracy : " + str(len(correct_indices)/(len(predicted_classes))))
model.summary()
