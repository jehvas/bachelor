from utility.undersample_split import resize_under_sample
from utility.utility import file_exists, load


def check_mini_load(dataset, dataset_mode, dimensionCount):
    feature_file_name = dataset_mode + "/" + dataset.get_name() + '_features_' + str(dimensionCount)
    if file_exists(dataset.get_name() + "_saved_labels") and file_exists(feature_file_name):
        print('Using mini loading')
        labels = load(dataset.get_name() + "_saved_labels")
        if dataset_mode == "2000":
            emails, labels = resize_under_sample(labels, labels)
        return True, labels
    return False, None
