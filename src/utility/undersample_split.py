import random
from collections import Counter

import numpy as np


def under_sample_split(features, labels, test_size=0.2, random_state=None):
    if len(features) != len(labels):
        raise Exception(
            "Features not equal labels length: len(features): {}, len(labels): {}".format(len(features), len(labels)))
    if random_state:
        random.seed(random_state)
    comb = list(zip(features, labels))
    random.shuffle(comb)
    features, labels = zip(*comb)
    label_counter = list(Counter(labels).values())
    min_class = min(label_counter)
    train_amount = int(min_class * (1 - test_size))  # Number of train data per class
    class_counts = [0] * len(label_counter)  # Init empty label counter
    x_train, y_train, x_test, y_test = [], [], [], []
    for i in range(len(labels) - 1, -1, -1):
        label = labels[i]
        if class_counts[label] < train_amount:
            class_counts[label] += 1
            x_train.append(features[i])
            y_train.append(label)
        else:
            x_test.append(features[i])
            y_test.append(label)
    print('train:', Counter(y_train))
    print('test:', Counter(y_test))
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def resize_under_sample(emails, labels, size=1000):
    class_count = len(set(labels))
    num_per_class = size/class_count

    class_counts = [0] * class_count
    re_emails, re_labels = [], []
    for i in range(0, len(labels)):
        label = labels[i]
        if class_counts[label] < num_per_class:
            class_counts[label] += 1
            re_labels.append(label)
            re_emails.append(emails[i])
    return re_emails, re_labels