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
                labels = list((list(zip(*labels)))[label_idx])
                if label_idx == 1:
                    return reviews, self.filter_gender(labels)
                return reviews, labels

        direc = ROOTPATH + "/data/Trustpilot/denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp"

        with open(direc, 'r', encoding='UTF8') as f:
            small_file = []
            for i, l in enumerate(f):
                small_file.append(l)
                #if len(small_file) == 2500:
                    #break
            val = Parallel(n_jobs=-1)(delayed(self.process_user_object)(i, line) for i, line in enumerate(small_file))
            reviews = []
            ratings = []
            num_empty_reviews = 0
            genders = []
            num_in_class = [0] * 5
            for data in val:
                _reviews, _ratings, _num_empty_reviews, _gender = data

                for i, rating in enumerate(_ratings):
                    try:
                        if num_in_class[rating-1] <= 10000:
                            num_in_class[rating-1] += 1
                            reviews.append(_reviews[i])
                            ratings.append(_ratings[i]-1) # to make sure ratings match index starting from 0
                            genders.append(_gender)
                    except:
                        print(rating)

                num_empty_reviews += _num_empty_reviews


        labels = list(zip(ratings, genders))
        labels = np.asarray(labels)
        reviews = np.asarray(reviews)
        super().post_load(reviews, labels)
        print(Counter(ratings))
        print(Counter(genders))
        print("Empty reviews: ", num_empty_reviews)
        labels = list((list(zip(*labels)))[label_idx])
        if label_idx == 1:
            print(len(reviews), " , ", len(genders))
            return self.filter_genders(reviews, genders)
        print(len(reviews), " , ", len(labels))
        return reviews, labels

    def filter_genders(self, reviews, genders):
        idx_to_delete = []
        for i, gender in enumerate(genders):
            if gender == "U":
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
                        gender = user_json_object["gender"]
                    else:
                        gender = "U"
                else:
                    num_empty_review += 1
        return reviews, ratings, num_empty_review, gender
