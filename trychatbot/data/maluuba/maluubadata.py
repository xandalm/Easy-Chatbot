
import pandas as pd
import itertools

import os

from ..prepare_data import Data

from six.moves import cPickle

class MaluubaData(Data):

    def __init__(self):
        super().__init__()
        self.path = "data/maluuba/"
        # self.dataset_size = 10407
        self.dataset_size = 1000

    def _normalize_data(self):
        data = pd.read_json(self.path + "frames.json")
        data['chat'] = data['turns'].apply(lambda x: [item['text'] for item in x])
        data['user'] = data['chat'].apply(lambda x: x[0::2])
        data['bot'] = data['chat'].apply(lambda x: x[1::2])
        dataset = data.apply(lambda x: list(itertools.zip_longest(x['user'], x['bot'], fillvalue='')), axis=1)
        all_convos = []
        _ = dataset.apply(lambda x: all_convos.extend(x))
        cPickle.dump(all_convos, open(self.path + "all_convos.pkl","wb"))