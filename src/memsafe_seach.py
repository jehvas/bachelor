import os

import sys

algorithms_to_use = sys.argv[1].lower()
datasets_to_use = sys.argv[2].lower()
amount = int(sys.argv[3])

while True:
    os.system('python parameter_search.py {} {} {}'.format(algorithms_to_use, datasets_to_use, amount))
