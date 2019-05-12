import math
import random
from typing import Dict

from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.proximal_adagrad import ProximalAdagradOptimizer
from tensorflow.python.training.proximal_gradient_descent import ProximalGradientDescentOptimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer


def get_random_params(algorithm, input_dim, output_dim):
    if algorithm == 'RNN_Tensorflow' or algorithm == 'MLP_Tensorflow' or algorithm == 'Bi_LSTM_Tensorflow':
        # layer_dim = 5 - int(math.log10(random.randint(10, 9000)))
        hidden_dim = random.randint(10, 500)
        optimizer, lr = pick_optimizer()
        params = {
            'hidden_dim': hidden_dim,
            # 'layer_dim': layer_dim,
            'input_function': pick_random_activation_function(),
            # 'hidden_layers': generate_middle_layers(layer_dim, algorithm),
            'output_function': pick_random_activation_function(),
            'optimizer': optimizer,
            'learning_rate': lr,
            'output_dim': output_dim,
            'input_dim': input_dim,
        }
        if algorithm == "MLP_Tensorflow":
            params["hidden_layers"] = [("Dense", 92, "relu"),
                                       #("LeakyReLU", "", ""),
                                       ("Dropout", 0.3, ""),
                                       ("Dense", output_dim, 'softmax')]
        elif algorithm == "RNN_Tensorflow":
            params["hidden_layers"] = [("RNN", 240, "tanh"),
                                       #("LeakyReLU", "", ""),
                                       ("Dropout", 0.0, ""),
                                       ("RNN", 148, "relu"),
                                       #("LeakyReLU", "", ""),
                                       ("Dropout", 0.5, ""),
                                       ("Dense", 300, "linear"),
                                       ("LeakyReLU", "", ""),
                                       ("Dropout", 0.5, ""),
                                       ("Dense", output_dim, 'softmax')]
        elif algorithm == "Bi_LSTM_Tensorflow":
            params["hidden_layers"] = [("Bi_LSTM", 14, "softmax"),
                                       #("LeakyReLU", "", "")   ,
                                       ("Dropout", 0.3, ""),
                                       ("Bi_LSTM", 19, "relu"),
                                       #("LeakyReLU", "", ""),
                                       ("Dropout", 0.3, ""),
                                       ("Dense", 300, "linear"),
                                       ("LeakyReLU", "", ""),
                                       ("Dropout", 0.3, ""),
                                       ("Dense", output_dim, 'softmax')]
        return params

    elif algorithm == 'SVM':
        return {
            'loss_function': random.choice(["hinge", "squared_hinge"]),
            'class_weights': random.choice(["balanced"]),
            'penalty': random.choice(["l2"])
        }
    elif algorithm == 'Perceptron':
        return {
            'class_weights': random.choice(["balanced"]),
            'penalty': random.choice(["elasticnet", None, "l2", "l1"])
        }


def pick_random_activation_function():
    possible_activations = [
        "relu",
        "softmax",
        # "sigmoid",
        # "elu",
        # "selu",
        # "softplus",
        # "softsign",
        "tanh"
    ]
    return random.choice(possible_activations)


def pick_optimizer():
    random_lr = random.randint(1, 1000) / 10000
    possible_optimizers = [
        # AdagradOptimizer(learning_rate=random_lr),
        SGD(lr=0.0398, decay=1e-6),
        # AdadeltaOptimizer(learning_rate=random_lr),
        #Adam(lr=0.0311, decay=1e-6),
        # FtrlOptimizer(learning_rate=random_lr),
        # ProximalAdagradOptimizer(learning_rate=random_lr),
        # ProximalGradientDescentOptimizer(learning_rate=random_lr),
        # RMSPropOptimizer(learning_rate=random_lr)
    ]
    optimizer_to_return = random.choice(possible_optimizers)
    return optimizer_to_return, str(random_lr)


def pick_random_class_weights(num_labels):
    classes = [i for i in range(num_labels)]
    class_weight_dic = {}
    for i in classes:
        class_weight_dic[i] = random.randint(1, 100)
    return class_weight_dic