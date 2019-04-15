import datetime
import numpy as np
import pickle as p

import os

from keras.optimizers import Optimizer

from rootfile import ROOTPATH

output_path = ROOTPATH + "output/"


def save(object_to_save, file_name):
    file_name = output_path + file_name
    with open(file_name, "wb") as fp:
        p.dump(object_to_save, fp)
    print("Utility saved", file_name)


def file_exists(file_name):
    return os.path.exists(output_path + file_name)


def get_file_path(file_name):
    return output_path + file_name


def load(file_name):
    file_name = output_path + file_name
    with open(file_name, "rb") as fp:
        data = p.load(fp)
    print("Utility loaded", file_name)
    return data


def print_progress(progress, total):
    print("{:.2f}".format(progress / total * 100), "%")


def log_to_file(parameters, fscore, file_path, time_taken):
    create_file_is_not_exists(file_path, parameters)
    avg = sum(fscore) / len(fscore)
    with open(file_path, 'a+') as f:
        f.write(str(avg) + ", ")
        for key, value in parameters.items():
            if isinstance(value, (np.ndarray, np.generic)):
                f.write(np.array2string(value, separator=';', max_line_width=500) + ", ")
            elif type(value) is dict:
                f.write(';'.join([str(k2)+":"+str(v2) for k2, v2 in value.items()]) + ", ")
            elif isinstance(value, Optimizer):
                f.write(value.lr.name[:-5] + ", ")
            else:
                f.write(str(value) + ", ")
        f.write(np.array2string(fscore, separator=';', max_line_width=500) + ", ")
        f.write(str(time_taken) + ",")
        f.write("\n")


def create_file_is_not_exists(file_path, parameters):
    if not os.path.isfile(file_path):
        header_info = ["Avg_Fscore"]
        for key, value in parameters.items():
            header_info += [key]
        header_info += ["Fscore"]
        header_info += ["Time_taken"]
        with open(file_path, 'w+') as f:
            f.write(','.join(header_info) + '\n')


def setup_result_folder(algorithm_name, dataset_name,):
    if not os.path.exists(ROOTPATH + "Results/" + algorithm_name):
        os.mkdir(ROOTPATH + "Results/" + algorithm_name)
    if not os.path.exists(ROOTPATH + "Results/" + algorithm_name + "/" + dataset_name):
        os.mkdir(ROOTPATH + "Results/" + algorithm_name + "/" + dataset_name)