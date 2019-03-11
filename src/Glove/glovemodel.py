import numpy as np
import os
from utility.utility import print_progress


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile, 'r+', encoding="utf8") as f:
        # total of 1917494 lines in glove.42B.300d.txt
        total = os.stat(gloveFile).st_size
        c = 0
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
            if (c % 50000 == 0):
                print_progress(c, total)
            c = c + len(line)
        print("c: ", c, "total: ", total)
        print("Done.", len(model), " words of loaded!")
        return model


#m = loadGloveModel("glove.42B.300d.txt")


