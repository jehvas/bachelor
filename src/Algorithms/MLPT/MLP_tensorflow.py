from typing import List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts
from tensorflow import keras


def get_name():
    return 'MLP_Tensorflow'


def run_train(dataset, features, labels, parameters) -> (List, List, List):
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    output_dim = parameters['output_dim']
    hidden_dim = parameters['hidden_dim']
    input_dim = parameters['input_dim']

    model = keras.Sequential([
        keras.layers.Dense(hidden_dim, activation=tf.nn.relu, input_shape=(input_dim,)),
        keras.layers.Dense(hidden_dim, activation=tf.nn.relu),
        keras.layers.Dense(output_dim, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, np.asarray(y_train), epochs=20)

    predictions = model.predict(x_test)
    predictions = [np.argmax(x) for x in predictions]

    return [], y_test, predictions
