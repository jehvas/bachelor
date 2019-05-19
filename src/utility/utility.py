import os
import pickle as p

import errno
import numpy as np
from tensorflow.python.keras.optimizers import Optimizer
from rootfile import ROOTPATH

output_path = ROOTPATH + "output/"


def check_directory(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def save(object_to_save, file_name, verbose=False):
    file_name = output_path + file_name
    check_directory(file_name)
    with open(file_name, "wb") as fp:
        p.dump(object_to_save, fp)
    if verbose:
        print("Utility saved", file_name)


def file_exists(file_name):
    return os.path.exists(output_path + file_name)


def get_file_path(file_name):
    return output_path + file_name


def load(file_name, verbose=False):
    file_name = output_path + file_name
    with open(file_name, "rb") as fp:
        data = p.load(fp)
    if verbose:
        print("Utility loaded", file_name)
    return data


def print_progress(progress, total):
    if progress % int(total / 10) == 0:
        print("{:.2f}".format(progress / total * 100), "%")


def log_to_file(parameters, fscore, file_path, time_taken, guid):
    tmp_param = {}
    for key in sorted(parameters):
        tmp_param[key] = parameters[key]
    parameters = tmp_param
    create_file_is_not_exists(file_path, parameters)
    avg = sum(fscore) / len(fscore)
    with open(file_path, 'a+') as f:
        f.write(str(avg).replace('.', ',') + "\t ")
        for key, value in parameters.items():
            if isinstance(value, (np.ndarray, np.generic)):
                f.write(np.array2string(value, separator=';', max_line_width=500) + "\t ")
            elif type(value) is dict:
                f.write(';'.join([str(k2) + ":" + str(v2) for k2, v2 in value.items()]) + "\t ")
            elif isinstance(value, Optimizer):
                f.write(value.lr._shared_name + "\t ")
            elif type(value) is list:
                if type(value[0]) is tuple:
                    f.write(";".join([str(tup) for tup in value]) + "\t ")
                else:
                    f.write(';'.join([str(v) for v in value]) + "\t ")
            else:
                f.write(str(value) + "\t ")
        f.write(np.array2string(fscore, separator=';', max_line_width=500) + "\t ")
        f.write(str(time_taken) + "\t")
        f.write(guid + "\t")
        f.write("\n")


def create_file_is_not_exists(file_path, parameters):
    if not os.path.isfile(file_path):
        header_info = ["Avg_Fscore"]
        for key, value in parameters.items():
            header_info += [key]
        header_info += ["Fscore"]
        header_info += ["Time_taken"]
        header_info += ['GUID']
        with open(file_path, 'w+') as f:
            f.write('\t'.join(header_info) + '\n')


def setup_result_folder(algorithm_name, dataset_name):
    if not os.path.exists(ROOTPATH + "Results/" + algorithm_name):
        os.mkdir(ROOTPATH + "Results/" + algorithm_name)
    if not os.path.exists(ROOTPATH + "Results/" + algorithm_name + "/" + dataset_name):
        os.mkdir(ROOTPATH + "Results/" + algorithm_name + "/" + dataset_name)
