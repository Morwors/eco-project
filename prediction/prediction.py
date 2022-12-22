import cv2
import keras.layers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from keras.utils import plot_model
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set the number of past arrays to use for prediction
num_arrays = 10

# Set the size of the arrays
array_size = (128, 128)



def teachModel(arrays, country):
    try:
        model = tf.keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(arrays[0].shape[1], input_dim=arrays[0].shape[1]),
            keras.layers.Dense(arrays[0].shape[1] * 2, input_dim=arrays[0].shape[1]),
            keras.layers.Dense(arrays[0].shape[1], input_dim=arrays[0].shape[1]),
        ])

        for i in range(len(arrays) - 1):
            x = np.round(arrays[i])
            y = np.round(arrays[i + 1])
            model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy', 'mse'])
            history = model.fit(x, y, epochs=1000)
            plot_history(history)

        model.save("demo/"+country+"/model/vegetation.h5")
        return True
    except tf.errors.FAILED_PRECONDITION:
        raise tf.errors.FAILED_PRECONDITION
    except:
        return False

def plot_history(history):
    print("Plotting history with: ", history.history)
    plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['mean_squared_error'], label="inacuracy")
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_prediction(test_labels, test_predictions):
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')
    print(error)
    plt.show()


def predictFutureVegetation(inputImgArr, country):
    try:
        model = tf.keras.models.load_model('demo/'+country+'/model/vegetation.h5')
        result = model.predict(inputImgArr)
        result = np.round(result * 255.0)
        result[result > 255.0] = 255.0
        cv2.imwrite('demo/'+country+'/whitemap/prediction.jpg', result)
        return True
    except:
        return False

def predict():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    y = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)

    model = tf.keras.Sequential([
        keras.layers.Dense(units=3, input_shape=(3,)),
        keras.layers.Dense(units=3)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_squared_error '])

    model.fit(x, y, epochs=2000)
    loss, acc = model.evaluate(x, y, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    result = model.predict(x)
    print(result)

