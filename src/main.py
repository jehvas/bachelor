import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from Glove.glovemodel import GloVe
from utility.Parameters import get_params
from utility.argument_parser import parse_arguments

algorithms_to_use, datasets_to_use, amount, dataset_mode = parse_arguments(sys.argv)

for dataset in datasets_to_use:
    emails, labels = dataset.load()
    glove = GloVe(300)

    features = glove.get_features(emails, dataset, dataset_mode)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        parameters = get_params(algorithm.get_name(), dataset)
        print(str(parameters))

        assert not np.any(np.isnan(features))
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1, stratify=labels)

        start_time = time.time()
        algorithm.run_train(dataset, x_train, y_train, x_test, y_test, labels, parameters)
        time_taken = time.time() - start_time
        print("Finished in {:.3f}".format(time_taken))

