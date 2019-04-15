import random
import time

from sklearn.metrics import precision_recall_fscore_support
from Algorithms.SVM import SVM
from DatasetsConsumers.Newsgroups import Newsgroups
from DatasetsConsumers.Spamassassin import Spamassassin
from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Random_Parameters import get_random_params
from utility.utility import log_to_file


def generate_middle_layers(num_layers):
    """
    Generate layers that are randomly filled with dropout layers.
    Returns: List of tuple (layer_type, parameter)
    Parameter is ether an activation function for the hidden layer, or a dropout percentage for the dropout layer
    """
    layers = []
    for i in range(num_layers):
        dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
        if dropout_chance > 0:
            layers.append(('dropout', dropout_chance))
        layers.append(('hidden', pick_random_activation_function()))
    dropout_chance = int(random.randint(1, 2) / 2) * random.randint(1, 80) / 100  # 50% chance to be 0
    if dropout_chance > 0:
        layers.append(('dropout', dropout_chance))
    return layers
    '''possible_layers = [tf.keras.layers.LeakyReLU(dim),
                       tf.keras.layers.ELU(dim),
                       tf.keras.layers.ReLU(random.randint(1, 100) / 100,
                                            random.randint(1, 100) / 100,
                                            random.randint(1, 50)),
                       # tf.keras.layers.Softmax(random.randint(-2, 2)),
                       tf.keras.layers.Dense(dim, activation=pick_activation_function())
                       ]
    return [possible_layers[random.randint(0, len(possible_layers) - 1)] for _ in range(num_layers)]'''


def pick_random_activation_function():
    possible_activations = ["relu", "softmax", "sigmoid", "elu", "selu", "softplus",
                            "softsign", "tanh"]
    return possible_activations[random.randint(0, len(possible_activations) - 1)]


counter = 1
dataset_consumer = Newsgroups()
algorithm = SVM

emails, labels = dataset_consumer.load(True)
glove = GloVe(200)
features = glove.get_features(emails, dataset_consumer)
print("Running algorithm:", algorithm.get_name())
while True:
    output_dim = len(set(labels))
    parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)

    print("\n#### STARTING RUN NUMBER {} #####\n".format(counter))
    print(str(parameters))
    start_time = time.time()
    data_to_plot, y_test, rounded_predictions = algorithm.run_train(dataset_consumer, features, labels,
                                                                    parameters)
    time_taken = time.time() - start_time
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, rounded_predictions)
    # print("\nPrecision: ", precision)
    # print("\nRecall: ", recall)
    # print("\nFscore: ", fscore)
    # print("\n")
    print("Avg fScore:", (sum(fscore) / len(fscore)))
    file_path = ROOTPATH + "Results/" + algorithm.get_name() + "_" + dataset_consumer.get_name() + "_resultsfile.csv"
    log_to_file(parameters, fscore, file_path, time_taken)
    counter += 1
