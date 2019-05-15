from tensorflow.python.keras.optimizers import Adam, SGD


def leaky_to_linear(parameters):
    if 'hidden_layers' in parameters:
        for i, (layer_type, num, activation) in enumerate(parameters['hidden_layers']):
            new_activation = 'linear' if activation.lower() == 'leakyrelu' else activation
            parameters['hidden_layers'][i] = (layer_type, num, new_activation)
    return parameters


def get_params(algorithm, dataset):
    if algorithm == 'RNN_Tensorflow':
        if dataset.get_name() == "Spamassassin":
            return {
                'hidden_layers': [("RNN", 207, "LeakyReLU"),
                                  ("Dropout", 0.4, ""),
                                  ("RNN", 161, "LeakyReLU"),
                                  ("Dropout", 0.3, ""),
                                  ("Dense", 99, "LeakyReLU"),
                                  ("Dropout", 0.1, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0843),
                'learning_rate': '0.0843'}
        if dataset.get_name() == "Newsgroups":
            return {
                'hidden_layers': [("RNN", 236, "LeakyReLU"),
                                  ("Dropout", 0.5, ""),
                                  ("RNN", 233, "LeakyReLU"),
                                  ("Dropout", 0.2, ""),
                                  ("Dense", 138, "LeakyReLU"),
                                  ("Dropout", 0.2, ""),
                                  ("Dense", 20, "softmax")],
                'optimizer': SGD(lr=0.0520),
                'learning_rate': '0.0520'}
        if dataset.get_name() == "EnronEvidence":
            return {
                'hidden_layers': [("RNN", 63, "softmax"),
                                  ("Dropout", 0.1, ""),
                                  ("RNN", 173, "relu"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 147, "relu"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0614),
                'learning_rate': '0.0614'}
        if dataset.get_name() == "EnronFinancial":
            return {
                'hidden_layers': [("RNN", 80, "relu"),
                                  ("Dropout", 0.1, ""),
                                  ("RNN", 16, "LeakyReLU"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 193, "relu"),
                                  ("Dropout", 0.1, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0316),
                'learning_rate': '0.0316'}
        if dataset.get_name() == "Trustpilot":
            return {
                'hidden_layers': [("RNN", 238, "linear"),
                                  ("Dropout", 0.3, ""),
                                  ("RNN", 159, "linear"),
                                  ("Dropout", 0.0, ""),
                                  ("Dense", 105, "linear"),
                                  ("Dropout", 0.1, ""),
                                  ("Dense", 5, "softmax")],
                'optimizer': SGD(lr=0.0464),
                'learning_rate': '0.0464'}

    elif algorithm == 'MLP_Tensorflow':
        if dataset.get_name() == "Spamassassin":
            return {
                'hidden_layers': [("Dense", 265, "relu"),
                                  ("Dropout", 0.3, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': Adam(lr=0.0311, decay=1e-6),
                'learning_rate': '0.0311'}
        if dataset.get_name() == "Newsgroups":
            return {
                'hidden_layers': [("Dense", 259, "relu"),
                                  ("Dropout", 0.1, ""),
                                  ("Dense", 20, "softmax")],
                'optimizer': SGD(lr=0.0681, decay=1e-6),
                'learning_rate': '0.0681'}
        if dataset.get_name() == "EnronEvidence":
            return {
                'hidden_layers': [("Dense", 277, "relu"),
                                  ("Dropout", 0.3, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0211, decay=1e-6),
                'learning_rate': '0.0211'}
        if dataset.get_name() == "EnronFinancial":
            return {
                'hidden_layers': [("Dense", 237, "relu"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0399, decay=1e-6),
                'learning_rate': '0.0399'}
        if dataset.get_name() == "Trustpilot":
            return {
                'hidden_layers': [("Dense", 92, "relu"),
                                  ("Dropout", 0.3, ""),
                                  ("Dense", 5, "softmax")],
                'optimizer': SGD(lr=0.0404, decay=1e-6),
                'learning_rate': '0.0404'}

    elif algorithm == 'Bi_LSTM_Tensorflow':
        if dataset.get_name() == "Spamassassin":
            return {
                'hidden_layers': [("Bi_LSTM", 52, "softmax"),
                                  ("Dropout", 0.4, ""),
                                  ("Bi_LSTM", 122, "softmax"),
                                  ("Dropout", 0, ""),
                                  ("Dense", 26, "relu"),
                                  ("Dropout", 0.1, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0715, decay=1e-6),
                'learning_rate': '0.0715'}
        if dataset.get_name() == "Newsgroups":
            return {
                'hidden_layers': [("Bi_LSTM", 278, "softmax"),
                                  ("Dropout", 0.3, ""),
                                  ("Bi_LSTM", 277, "softmax"),
                                  ("Dropout", 0.2, ""),
                                  ("Dense", 12, "relu"),
                                  ("Dropout", 0.2, ""),
                                  ("Dense", 20, "softmax")],
                'optimizer': Adam(lr=0.0037, decay=1e-6),
                'learning_rate': '0.0037'}
        if dataset.get_name() == "EnronEvidence":
            return {
                'hidden_layers': [("Bi_LSTM", 225, "softmax"),
                                  ("Dropout", 0.2, ""),
                                  ("Bi_LSTM", 171, "softmax"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 25, "relu"),
                                  ("Dropout", 0.4, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0470, decay=1e-6),
                'learning_rate': '0.0470'}
        if dataset.get_name() == "EnronFinancial":
            return {
                'hidden_layers': [("Bi_LSTM", 206, "softmax"),
                                  ("Dropout", 0.2, ""),
                                  ("Bi_LSTM", 98, "softmax"),
                                  ("Dropout", 0.4, ""),
                                  ("Dense", 18, "relu"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 2, "softmax")],
                'optimizer': SGD(lr=0.0864, decay=1e-6),
                'learning_rate': '0.0864'}
        if dataset.get_name() == "Trustpilot":
            return {
                'hidden_layers': [("Bi_LSTM", 229, "relu"),
                                  ("Dropout", 0, ""),
                                  ("Bi_LSTM", 287, "softmax"),
                                  ("Dropout", 0.4, ""),
                                  ("Dense", 14, "softmax"),
                                  ("Dropout", 0.5, ""),
                                  ("Dense", 5, "softmax")],
                'optimizer': SGD(lr=0.0004, decay=1e-6),
                'learning_rate': '0.0004'}

    elif algorithm == 'SVM':
        if dataset.get_name() == 'Spamassassin':
            return {
                "loss_function": "hinge"
            }
        if dataset.get_name() == 'Newsgroups':
            return {
                "loss_function": "hinge"
            }
        if dataset.get_name() == 'EnronEvidence':
            return {
                "loss_function": "hinge"
            }
        if dataset.get_name() == 'EnronFinancial':
            return {
                "loss_function": "hinge"
            }
        if dataset.get_name() == 'Trustpilot':
            return {
                "loss_function": "hinge"
            }


    elif algorithm == 'Perceptron':
        if dataset.get_name() == 'Spamassassin':
            return {
                "penalty": None
            }
        if dataset.get_name() == 'Newsgroups':
            return {
                "penalty": "l1"
            }
        if dataset.get_name() == 'EnronEvidence':
            return {
                "penalty": "l1"
            }
        if dataset.get_name() == 'EnronFinancial':
            return {
                "penalty": None
            }
        if dataset.get_name() == 'Trustpilot':
            return {
                "penalty": "l1"
            }
