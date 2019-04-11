import ast
from collections import Counter
from typing import List
import numpy as np
from joblib import Parallel, delayed

from DatasetsConsumers.AbstractDataset import AbstractDataset


class Trustpilot(AbstractDataset):
    num_no_gender_specified = 0
    num_json_parse_errors = 0
    gender_list: List = []
    num_line = 0

    def load(self, load_filtered_data=False, label_idx=0):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                reviews, labels = load_check_result
                ratings, genders = labels
                if label_idx == 1:
                    return self.filter_gender(reviews, genders)
                self.classes = [0, 1, 2, 3, 4]
                return reviews, ratings

        direc = "../data/Trustpilot/united_states.auto-adjusted_gender.geocoded.jsonl.tmp"

        with open(direc, 'r', encoding='UTF8') as f:
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
            num_empty_reviews = sum(num_empty_reviews)
            num_no_rating = sum(num_no_rating)
            print(num_empty_reviews, num_no_rating)

            self.classes = [0, 1, 2, 3, 4]

            ratings = np.array(ratings)
            genders = np.array(genders)
            reviews = np.array(reviews)

            super().post_load(reviews, (ratings, genders))

            print(Counter(ratings))
            print(Counter(genders))
            print("Empty reviews: ", num_empty_reviews)
            # labels = list((list(zip(*labels)))[label_idx])
            if label_idx == 1:
                return self.filter_genders(reviews, genders)

            return reviews, ratings

    def filter_genders(self, reviews: np.array, genders: np.array) -> (np.array, np.array):
        idx_to_delete: List = []
        for i, gender in enumerate(genders):
            if gender == 2:
                idx_to_delete += i
        for i in reversed(idx_to_delete):
            genders.pop(i)
            reviews.pop(i)
        return reviews, genders

    def process_user_object(self, i, line):
        if i % 10000 == 0:
            print(i)
        user_json_object = ast.literal_eval(line)
        if "reviews" in user_json_object:
            gender = 2
            reviews: List = []
            ratings: List = []
            genders: List = []
            num_empty_review = 0
            num_empty_rating= 0
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
