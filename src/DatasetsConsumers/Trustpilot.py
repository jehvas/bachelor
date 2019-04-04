import os
from collections import Counter
import json
import numpy as np
from flashtext.keyword import KeywordProcessor
from flask import Flask
from joblib import Parallel, delayed

from DatasetsConsumers.AbstractDataset import AbstractDataset
from rootfile import ROOTPATH
import ast


class Trustpilot(AbstractDataset):
    num_no_gender_specified = 0
    num_json_parse_errors = 0
    gender_list = []
    num_line = 0
    keyword_processor = KeywordProcessor()

    def load(self, load_filtered_data=False, label_idx=0):
        if load_filtered_data:
            load_check_result = super().pre_load()
            if load_check_result is not None:
                reviews, labels = load_check_result
                ratings, genders = labels
                if label_idx == 1:
                    return self.filter_gender(reviews, genders)
                return reviews, ratings

        direc = ROOTPATH + "/data/Trustpilot/united_states.auto-adjusted_gender.geocoded.jsonl.tmp"

        with open(direc, 'r', encoding='UTF8') as f:
            small_file = []
            for i, l in enumerate(f):
                small_file.append(l)
                if len(small_file) == 2000:
                    break
            val = Parallel(n_jobs=1)(delayed(self.process_user_object)(i, line) for i, line in enumerate(small_file))
            reviews = []
            ratings = []
            num_empty_reviews = 0
            genders = []
            num_in_class = [0] * 5
            for data in val:
                _reviews, _ratings, _num_empty_reviews, _gender = data

                for i, rating in enumerate(_ratings):
                    rating_num = int(rating)-1 # to make sure ratings match index starting from 0 and to accommodate that rating might be a string.

                    if num_in_class[rating_num] < 100:
                        num_in_class[rating_num] += 1
                        reviews.append(_reviews[i])
                        ratings.append(rating_num)
                        genders.append(_gender)

                num_empty_reviews += _num_empty_reviews



        ratings = np.array(ratings)
        genders = np.array(genders)
        reviews = np.array(reviews)
        super().post_load(reviews, (ratings, genders))

        print(Counter(ratings))
        print(Counter(genders))
        print("Empty reviews: ", num_empty_reviews)
        #labels = list((list(zip(*labels)))[label_idx])
        if label_idx == 1:
            return self.filter_genders(reviews, genders)

        return reviews, ratings

    def filter_genders(self, reviews, genders):
        idx_to_delete = []
        for i, gender in enumerate(genders):
            if gender == 2:
                idx_to_delete += i
        for i in reversed(idx_to_delete):
            genders.pop(i)
            reviews.pop(i)
        return reviews, genders

    def process_user_object(self, i, line):
        if i % 5000 == 0:
            print(i)
        user_json_object = ast.literal_eval(line)
        reviews = []
        ratings = []
        num_empty_review = 0
        gender = ""
        if "reviews" in user_json_object:
            reviews_json_list = user_json_object["reviews"]
            for review in reviews_json_list:
                rating = review["rating"]
                text = review["text"]
                if len(text) != 0:
                    reviews.append(self.process_single_mail(text[0]))
                    ratings.append(rating)
                    if "gender" in user_json_object and (
                            user_json_object["gender"] == "M" or user_json_object["gender"] == "F"):
                        if user_json_object["gender"] == "M":
                            gender = 0
                        elif user_json_object["gender"] == "F":
                            gender = 1
                    else:
                        gender = 2
                else:
                    num_empty_review += 1
        return reviews, ratings, num_empty_review, gender
