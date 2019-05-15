import sys

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import set_random_seed

from Glove.glovemodel import GloVe
from utility.Parameters import leaky_to_linear, get_params
from utility.argument_parser import parse_arguments
from utility.minimal_loader import check_mini_load

algorithms_to_use, datasets_to_use, amount, dataset_mode = parse_arguments(sys.argv)

for dataset in datasets_to_use:
    dataset.mode = dataset_mode
    is_mini, mini_labels = check_mini_load(dataset, dataset_mode, 300)
    if is_mini:
        labels = mini_labels
        dataset.set_classes()
        emails = None
    else:
        emails, labels = dataset.load(dataset_mode=dataset_mode)
    glove = GloVe(300)

    features = glove.get_features(emails, dataset)

    for algorithm in algorithms_to_use:
        np.random.seed(1)
        set_random_seed(1)
        print("Running algorithm:", algorithm.get_name())
        parameters = leaky_to_linear(get_params(algorithm.get_name(), dataset))
        print(str(parameters))

        assert not np.any(np.isnan(features))
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1,
                                                            stratify=labels)

        algorithm.run_train(dataset, x_train, y_train, x_test, y_test, parameters, should_plot=True)
