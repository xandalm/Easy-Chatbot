
import pandas as pd
import itertools
import nltk

import json
import re
import os

from ..prepare_data import Data

from six.moves import cPickle

class AlexaData(Data):

    def __init__(self):
        super().__init__()
        self.path = "data/alexa/"
        self.dataset_size = 10000
    
    def _normalize_data(self):
        file = json.load(open(self.path + "train.json","r"))
        contents = []
        for i in file.items():
            contents.append(file[i[0]])
        data = pd.DataFrame(contents)
        data['chat'] = data["content"].apply(lambda x: [item["message"] for item in x])
        data['input'] = data['chat'].apply(lambda x: x[0::2])
        data['target'] = data['chat'].apply(lambda x: x[1::2])
        dataset = data.apply(lambda x: list(itertools.zip_longest(x['input'], x['target'], fillvalue='')), axis=1)
        all_convos = []
        _ = dataset.apply(lambda x: all_convos.extend(x))
        cPickle.dump(all_convos, open(self.path + "all_convos.pkl","wb"))