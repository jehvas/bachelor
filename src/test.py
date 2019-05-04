import random

import numpy as np
from tensorflow.python import set_random_seed

np.random.seed(1)
set_random_seed(1)
print(random.randint(1, 100))
