import time

import datetime
from sklearn.metrics import precision_recall_fscore_support

from Algorithms.MLPT import MLP_tensorflow
from DatasetsConsumers.Newsgroups import Newsgroups
# Load dataset
from Glove.glovemodel import GloVe
from utility.Parameters import get_params


def log_to_file(text):
    log_string = "{} {}".format(datetime.datetime.now(), text)
    print(log_string)


dataset_consumer = Newsgroups()
algorithm = MLP_tensorflow

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    parameters = {
        'batch_size': 128,
        'num_epochs': 1,
        'hidden_dim': 128,
        'layer_dim': 1,
        'learning_rate': 0.01,
        'class_weights': None,
        'dropout': 0.5,
        'max_len': 1024,
        'output_dim': len(set(labels)),
        'input_dim': features.shape[1]
    }

    log_to_file(str(parameters))

    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters, None, None,
                                                                    emails)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    print("\nPrecision: ", precision)
    print("\nRecall: ", recall)
    print("\nFscore: ", fscore)
    print("\n")
    print("Avg fScore:", (sum(fscore) / len(fscore)))
