from DatasetsConsumers.Chromium import Chromium
from DatasetsConsumers.Enron import Enron
from DatasetsConsumers.GoLang import GoLang

if __name__ == '__main__':
    emails, labels = Enron().load(True)
    print(emails)

'''
import os
import utility
from src.Algorithms.SVM.Algorithms.SVM import process_single_mail, sum_vectors, root_mean_square, create_sentence_vector

svm = utility.load("text-classifier.mdl")

direc = "testerino/"
files = os.listdir(direc)
emails = [direc + email for email in files]
email_data = []
for file in emails:
    print(file)
    with open(file, encoding="latin-1") as f:
        text = f.read()
        processed_email = process_single_mail(text)
        if len(processed_email) > 0:
            email_data.append(processed_email)

sum_vectors_array = sum_vectors(email_data)
rms_array = root_mean_square(sum_vectors_array)
features = create_sentence_vector(rms_array, sum_vectors_array)
for feature in features:
    res = svm.predict([feature])
    print(res)
'''
