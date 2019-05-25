import gc
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import set_random_seed, reset_default_graph, ops
from tensorflow.python.keras.backend import clear_session

from Glove.glovemodel import GloVe
from rootfile import ROOTPATH
from utility.Parameters import leaky_to_linear
from utility.Random_Parameters import get_random_params
from utility.argument_parser import parse_arguments
from utility.minimal_loader import check_mini_load
from utility.utility import setup_result_folder

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
        print("Running algorithm:", algorithm.get_name())
        plot_path = "{}Results/{}/{}/{}/".format(ROOTPATH, dataset_mode, algorithm.get_name(), dataset.get_name())
        if not os.path.exists(plot_path + "plots"):
            os.makedirs(plot_path + "plots")

        setup_result_folder(algorithm.get_name(), dataset.get_name())
        best_fscore = 0
        if os.path.exists(plot_path + "resultsfile.csv"):
            best_fscore = float(pd.read_csv(plot_path + "resultsfile.csv",
                                            delimiter='\t',
                                            index_col=False)['Avg_Fscore'].max().replace(',', '.'))
        print("best f-score on record {}".format(best_fscore))
        best_fscore_list = []
        output_dim = len(set(labels))
        assert not np.any(np.isnan(features))
        # Create training data
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1,
                                                            stratify=labels)

        print(Counter(y_train))
        for counter in range(1, (amount + 1)):
            np.random.seed(1)
            set_random_seed(1)
            print("#### STARTING RUN NUMBER {} #####".format(counter))
            parameters = leaky_to_linear(get_random_params(algorithm.get_name(), output_dim))
            print(str(parameters))
            try:
                algorithm.run_train(dataset, x_train, y_train, x_test, y_test, parameters, should_plot=False)
            except Exception as e:
                print("Caught exception: " + str(e))
                continue

            avg_fscore = np.average(algorithm.fscore)
            if avg_fscore > best_fscore:
                print('New champion! {} {}'.format(dataset, avg_fscore))
                best_fscore = avg_fscore
                algorithm.plot_data(dataset, y_test)

            clear_session()
            reset_default_graph()
            ops.reset_default_graph()
            gc.collect()
