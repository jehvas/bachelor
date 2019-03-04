import pickle as p


def save(objectToSave, name):
    print("Saving ", name)
    with open(name, 'wb') as fp:
        p.dump(objectToSave, fp)
    print("Done saving ", name)


def load(objectToLoad):
    print("Loading ", objectToLoad)
    with open(objectToLoad, "rb") as fp:
        svm = p.load(fp)
    print("Done loading " , objectToLoad)
    return svm
