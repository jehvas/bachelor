import os
import multiprocessing
from joblib import Parallel, delayed



def preprocess():
    direc = "../easy_ham/"
    files = os.listdir(direc)

    emails = [direc + email for email in files]

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(renameFIle)(email) for email in emails)


def renameFIle(email):



preprocess()
