import os
import random

import sys

import numpy as np

algorithms_to_use = sys.argv[1].lower()
datasets_to_use = sys.argv[2].lower()
amount = int(sys.argv[3])
mode = sys.argv[4]
while True:
    # os.system('python test.py')
    os.system('python parameter_search.py {} {} {} {}'.format(algorithms_to_use, datasets_to_use, amount, mode))
