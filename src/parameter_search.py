import os
import sys
import time
from collections import Counter

import gc
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import set_random_seed, reset_default_graph, ops
from tensorflow.python.keras.backend import clear_session

from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Random_Parameters import get_random_params
from utility.argument_parser import parse_arguments
from utility.undersample_split import resize_under_sample, under_sample_split
from utility.utility import log_to_file, setup_result_folder

algorithms_to_use, datasets_to_use, amount, dataset_mode = parse_arguments(sys.argv)
for dataset in datasets_to_use:
    emails, labels = dataset.load()
    if dataset_mode == "2000":
        emails, labels = resize_under_sample(emails, labels)

    glove = GloVe(300)

    # weights_matrix, features_from_matrix = glove.get_weights_matrix(emails, dataset, dataset_mode)
    features = glove.get_features(emails, dataset, dataset_mode)

    for algorithm in algorithms_to_use:
        print("Running algorithm:", algorithm.get_name())

        if not os.path.exists(ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots"):
            os.makedirs(ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/plots")

        setup_result_folder(algorithm.get_name(), dataset.get_name())
        best_fscore = 0
        best_fscore_list = []
        output_dim = len(set(labels))

        assert not np.any(np.isnan(features))

        # Create training data
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1, stratify=labels)

        print(Counter(y_train))
        for counter in range(1, (amount + 1)):
            np.random.seed(1)
            set_random_seed(1)
            print("#### STARTING RUN NUMBER {} #####".format(counter))

            parameters = get_random_params(algorithm.get_name(), features.shape[1], output_dim)

            print(str(parameters))

            start_time = time.time()
            try:
                algorithm.run_train(dataset, (x_train, y_train), (x_test, y_test), parameters)
            except Exception as e:
                print("Caught exception: " + str(e))
                continue

            avg_fscore = np.average(algorithm.fscore)
            if avg_fscore > best_fscore:
                print('New champion! {}'.format(avg_fscore))
                best_fscore = avg_fscore
                algorithm.plot_data(dataset, counter, dataset_mode, y_test)

            time_taken = time.time() - start_time

            file_path = ROOTPATH + "Results/" + dataset_mode + "/" + algorithm.get_name() + "/" + dataset.get_name() + "/"
            log_to_file(parameters, algorithm.fscore, file_path + "resultsfile.csv", time_taken, algorithm.guid)
            clear_session()
            reset_default_graph()
            ops.reset_default_graph()
            gc.collect()
