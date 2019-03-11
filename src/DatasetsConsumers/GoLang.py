import json

import os

from sklearn.externals.joblib import delayed, Parallel

from DatasetsConsumers import AbstractDataset
from DatasetsConsumers.CommonDevConsumer import CommonDevConsumer
from utility.utility import print_progress

#GO_LANG_JSON_PATH = '../../data/Go-nuts/golang-nuts.json/golang-nuts.json'
#GO_LANG_MAILS_PATH = '../../data/Go-nuts/golang-nuts/'

GO_LANG_JSON_PATH = '../../data/Chromium/chromium-dev.json/chromium-dev.json'
GO_LANG_MAILS_PATH = '../../data/Chromium/chromium-dev/chromium-dev/'


class GoLang(AbstractDataset.AbstractDataset):
    def load(self, load_filtered_data=False):
        commmon_l = CommonDevConsumer()
        return commmon_l.commonLoad('../../data/Go-nuts/golang-nuts.json/golang-nuts.json',
                                    '../../data/Go-nuts/golang-nuts/',
                                    load_filtered_data)

