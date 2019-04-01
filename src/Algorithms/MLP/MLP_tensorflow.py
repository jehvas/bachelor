import numpy as np

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow import keras
from sklearn.model_selection import train_test_split as tts
from DatasetsConsumers.Newsgroups import Newsgroups
from Glove.glovemodel import GloVe

if __name__ == '__main__':
    Dataset_Consumer = Newsgroups()
    emails, labels = Dataset_Consumer.load(True)

    # Load GloVe model
    GloVe_Obj = GloVe(50)
    features = GloVe_Obj.get_features(emails, Dataset_Consumer)

    # Create training data & SVM Stuff
    x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2, random_state=1, stratify=labels)

    def run():
        model = keras.Sequential([
            keras.layers.Dense(50, activation=tf.nn.relu),
            keras.layers.Dense(100, activation=tf.nn.relu),

            keras.layers.Dense(20, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, np.asarray(y_train), epochs=20)

        predictions = model.predict(x_test)
        predictions = [np.argmax(x) for x in predictions]

        precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predictions, average="micro")
        print("\n--- Results ---")
        print("Precision: ", precision)
        print("\n\nRecall: ", recall)
        print("\n\nF_score: ", fbeta_score)


    run()
