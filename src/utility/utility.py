import pickle as p

import os


output_path = "../../output/"


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
