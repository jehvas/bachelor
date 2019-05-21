import random

from tensorflow.python.keras.optimizers import Adam, SGD


def get_random_params(algorithm, output_dim):
    if algorithm == 'RNN_Tensorflow' or algorithm == 'MLP_Tensorflow' or algorithm == 'Bi_LSTM_Tensorflow':
        optimizer, lr = pick_optimizer()
        params = {
            'optimizer': optimizer,
            'learning_rate': lr,
        }
        if algorithm == "MLP_Tensorflow":
            params["hidden_layers"] = [("Dense", random.randint(10, 300), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", output_dim, 'softmax')]
        elif algorithm == "RNN_Tensorflow":
            params["hidden_layers"] = [("RNN", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("RNN", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", output_dim, 'softmax')]
        elif algorithm == "Bi_LSTM_Tensorflow":
            params["hidden_layers"] = [("Bi_LSTM", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Bi_LSTM", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
                                       ("Dense", random.randint(10, 600), pick_random_activation_function()),
                                       ("Dropout", random.randint(0, 5) / 10, ""),
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
        "LeakyReLU",
        # "sigmoid",
        # "elu",
        # "selu",
        # "softplus",
        # "softsign",
        # "tanh"
    ]
    return random.choice(possible_activations)


def pick_optimizer():
    random_lr = random.randint(1, 10000) / 100000
    possible_optimizers = [
        # AdagradOptimizer(learning_rate=random_lr),
        SGD(lr=random_lr, decay=1e-6),
        # AdadeltaOptimizer(learning_rate=random_lr),
        Adam(lr=random_lr, decay=1e-6),
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