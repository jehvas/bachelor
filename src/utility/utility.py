import pickle as p


def save(object_to_save, file_name):
    print("Saving ", file_name)
    with open(file_name, "wb") as fp:
        p.dump(object_to_save, fp)
    print("Done saving ", file_name)


def load(objectToLoad):
    print("Loading ", objectToLoad)
    with open(objectToLoad, "rb") as fp:
        svm = p.load(fp)
    print("Done loading ", objectToLoad)
    return svm


output_path = "../../output/"