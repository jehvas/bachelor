import ast
from collections import Counter
from typing import List
import numpy as np
from joblib import Parallel, delayed

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH


def filter_genders(reviews: np.array, genders: np.array):
    idx_to_delete: List = []
    for i, gender in enumerate(genders):
        if gender == 2:
            idx_to_delete += i
    for i in reversed(idx_to_delete):
        genders.pop(i)
        reviews.pop(i)
    return reviews, genders


class Trustpilot(AbstractDataset):
    num_no_gender_specified = 0
    num_json_parse_errors = 0
    gender_list: List = []
    num_line = 0
    num_lines = 0

    def sub_load(self, load_filtered_data=False, label_idx=0):
        directories = ROOTPATH + "data/Trustpilot/united_states.auto-adjusted_gender.geocoded.jsonl.tmp"

        self.num_lines = 0
        with open(directories, 'r', encoding='UTF8') as f:
            for _ in f:
                self.num_lines += 1
        with open(directories, 'r', encoding='UTF8') as f:
            val = Parallel(n_jobs=-1)(delayed(self.process_user_object)(i, line) for i, line in enumerate(f))
            reviews, ratings, genders, num_empty_reviews, num_no_rating = zip(*val)
            # Flatten lists
            reviews = [item for sublist in reviews for item in sublist]
            ratings = [item for sublist in ratings for item in sublist]
            genders = [item for sublist in genders for item in sublist]
            for i in range(len(reviews) - 1, -1, -1):
                if len(reviews[i]) == 0:
                    reviews.pop(i)
                    ratings.pop(i)
                    genders.pop(i)
            # num_empty_reviews = sum(num_empty_reviews)
            # num_no_rating = sum(num_no_rating)
            # print(num_empty_reviews, num_no_rating)
            # print("Empty reviews: ", num_empty_reviews)
            # labels = list((list(zip(*labels)))[label_idx])
            if label_idx == 1:
                return filter_genders(reviews, genders)

            return reviews, ratings

    def set_classes(self):
        self.classes = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star']

    def process_user_object(self, i, line):
        if i % int(self.num_lines / 10) == 0:
            print("{:.2f}%".format(i / (int(self.num_lines / 100) * 100) * 100))
        user_json_object = ast.literal_eval(line)
        if "reviews" in user_json_object:
            gender = 2
            reviews: List = []
            ratings: List = []
            genders: List = []
            num_empty_review = 0
            num_empty_rating = 0
            if "gender" in user_json_object:
                if user_json_object["gender"] == "M":
                    gender = 0
                elif user_json_object["gender"] == "F":
                    gender = 1
            for review in user_json_object["reviews"]:
                if review["rating"] is not None:
                    # 0-index ratings
                    rating = int(review["rating"]) - 1
                    text = review["text"]
                    if len(text) != 0:
                        reviews.append(self.process_single_mail(text[0]))
                        ratings.append(rating)
                        genders.append(gender)
                    else:
                        num_empty_review += 1
                else:
                    num_empty_rating += 1
            return reviews, ratings, genders, num_empty_review, num_empty_rating
