
import pandas as pd
import itertools
import nltk

import re
import os

from ..prepare_data import Data

from six.moves import cPickle

class PortuguesData(Data):

    def __init__(self):
        super().__init__()
        self.path = "data/portugues/"
        self.dataset_size = 10407

    def process_sentence(self, str, bot_input=False):
        tokens = nltk.word_tokenize(str)
        str = [t.lower() for t in tokens]
        if(bot_input):
            str = ["<START>"]+str[:self.MAX_LENGTH-1]+["<END>"]
        # else:
        #     str = str[:self.MAX_LENGTH]
        while True:
            try:
                str.remove("")
            except:
                break
        str = re.sub("\s+"," "," ".join(str)).strip()
        return str

    def _normalize_data(self):
        file = open(self.path + "conversations.txt","r",encoding="utf-8")
        data = ""
        try:
            data = data.join(line for line in file)
        except:
            pass
        all_convos = []
        for par in data.split("- -"):
            lines = par.split("  - ")
            input = lines[0].strip()
            target = "".join(l for l in lines[1:]).strip()
            t = (input, target)
            all_convos.append(t)
        cPickle.dump(all_convos, open(self.path + "all_convos.pkl","wb"))