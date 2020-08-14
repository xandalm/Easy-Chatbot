
import pandas as pd

from ..prepare_data import Data

from six.moves import cPickle

import itertools
import os
import ast

class CornellData(Data):

    def __init__(self):
        super().__init__()
        self.path = "data/cornell/"
        self.dataset_size = 20000
        
    def _normalize_data(self):
        MOVIE_LINES_FIELDS = ["lineID","characterID","movieID","character","text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID","character2ID","movieID","utteranceIDs"]
        self.__lines = {}
        self.__conversations = []
        self.__lines = self.__loadLines(os.path.join(self.path, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        self.__conversations = self.__loadConversations(os.path.join(self.path, "movie_conversations.txt"), MOVIE_CONVERSATIONS_FIELDS)
        self.__all_text_lines = []
        data = pd.DataFrame(self.__conversations)
        data['chat'] = data['lines'].apply(lambda x: [item['text'] for item in x])
        data['input'] = data['chat'].apply(lambda x: x[0::2])
        data['target'] = data['chat'].apply(lambda x: x[1::2])
        dataset = data.apply(lambda x: list(itertools.zip_longest(x['input'], x['target'], fillvalue='')), axis=1)
        all_convos = []
        _ = dataset.apply(lambda x: all_convos.extend(x))
        cPickle.dump(all_convos, open(self.path + "all_convos.pkl", "wb"))


    def __loadLines(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        lines = {}

        with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]

                lines[lineObj['lineID']] = lineObj

        return lines

    def __loadConversations(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        conversations = []

        with open(fileName, 'r', encoding='iso-8859-1') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.__lines[lineId])

                conversations.append(convObj)

        return conversations

