import numpy as np

from Glove.glovemodel import GloVe
from enums.enums import DatasetMode
from utility.undersample_split import resize_under_sample
from utility.utility import file_exists, load


def get_features_and_labels(dataset, dimension_count):
    is_mini, mini_labels = check_mini_load(dataset, dimension_count)
    if is_mini:
        labels = mini_labels
        dataset.set_classes()
        emails = None
    else:
        emails, labels = dataset.load()
    glove = GloVe(dimension_count)

    features = glove.get_features(emails, dataset)
    # features = glove.email_to_index_vector(emails)
    return features, labels


def check_mini_load(dataset, dimension_count):
    feature_file_name = "{}/{}_features_{}".format(dataset.mode, dataset.get_name(), dimension_count)
    if file_exists(dataset.get_name() + "_saved_labels") and file_exists(feature_file_name):
        # print('Using mini loading')
        labels = load(dataset.get_name() + "_saved_labels")
        if dataset.mode == DatasetMode.SIZE_2000:
            emails, labels = resize_under_sample(labels, labels)
        labels = np.asarray(labels)
        return True, labels
    return False, None
