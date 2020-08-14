import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer



from nltk.stem import PorterStemmer
from autocorrect import Speller

import unicodedata
import os
from six.moves import cPickle
import re
import time
import io

import tensorflow as tf
import tensorflow_addons as tfa

VOCAB_PATH = {
    'maluuba': "data/maluuba"
}

class Data:

    def __init__(self, data_origin):

        self.MAX_LENGTH = 25
        self.BATCH_SIZE = 64
        self.Dtype = tf.float32

        try:
            self.vocab = cPickle.load(open("data/"+data_origin+"/vocab","rb"))
        except:
            self.vocab = []
        
        self.vocab_size = len(self.vocab) + 1

        self.Tx = self.MAX_LENGTH
        self.Ty = self.MAX_LENGTH+1

        # word2index
        self.vocab_features_dict = dict([(token, i) for i, token in enumerate(self.vocab)])
        # index2word
        self.reverse_features_dict = dict((i, token) for token, i in self.vocab_features_dict.items())

    def __normalize_data(self):
        data = pd.read_json(self.path)
        data['chat'] = data['turns'].apply(lambda x: [item['text'] for item in x])
        data['user'] = data['chat'].apply(lambda x: x[0::2])
        data['bot'] = data['chat'].apply(lambda x: x[1::2])
        dataset = data.apply(lambda x: list(itertools.zip_longest(x['user'], x['bot'], fillvalue='')), axis=1)
        all_convos = []
        _ = dataset.apply(lambda x: all_convos.extend(x))
        cPickle.dump(all_convos, open("all_convos.pkl","wb"))

    
    def process_sentence(self, str, bot_input=False):
        
        str = str.strip().lower()
        str = re.sub(r"[^A-Za-z0-9(),!?\'\`:]"," ",str)
        str = re.sub(r"\'s"," \'s",str)
        str = re.sub(r"\'ve"," \'ve",str)
        str = re.sub(r"n\'t"," n\'t",str)
        str = re.sub(r"\'re"," \'re",str)
        str = re.sub(r"\'d"," \'d",str)
        str = re.sub(r"\'ll"," \'ll",str)
        str = re.sub(r","," , ",str)
        str = re.sub(r"!"," ! ",str)
        str = re.sub(r"\?"," ? ",str)
        str = re.sub(r"\s{2,}"," ",str)
        str = str.split(" ")
        # str = [re.sub(r"[0-9]+","_NUM",token) for token in str]
        # str = [self.stemmer.stem(re.sub(r'(.)\1+',r'\1\1',token)) for token in str]
        str = [self.spell(token).lower() for token in str]

        while True:
            try:
                str.remove("")
            except:
                break
        if(bot_input):
            str = str[:self.MAX_LENGTH-1]
            str.insert(0,"<START>")
            str.insert(len(str),"<END>")
        else:
            str = str[:self.MAX_LENGTH]
        # str = re.sub("\s+"," "," ".join(str)).strip()
        return str

    def __append_btw(self, str,elem):
        str.insert(len(str)-1,elem)

    def __append(self, str, elem):
        str.append(elem)

    def normalize_sentences_length(self, sentences, targetSentences=False):
        real_lens = [len(s) for s in sentences]
        bigger_sentence_size = max(real_lens)
        if (targetSentences):
            f = self.__append_btw
        else:
            f = self.__append
        for i in range(len(sentences)):
            str = sentences[i]
            for j in range((bigger_sentence_size) - len(str)):
                f(str," <PAD> ")
            sentences[i] = re.sub("\s+"," "," ".join(str)).strip()
        return list(zip(real_lens, sentences))

    def __create_docs(self):
        if(not os.path.isfile("all_convos.pkl")):
            self.__normalize_data()

        data = cPickle.load(open("all_convos.pkl","rb"))
        inputs = [item[0] for item in data]
        targets = [item[1] for item in data]

        if(os.path.isfile("inputs_processed.pkl")):
            inputs = cPickle.load(open("inputs_processed.pkl","rb"))
        else:
            print("init create docs")
            inputs = [self.process_sentence(item) for item in inputs]
            inputs = self.normalize_sentences_length(inputs)
            cPickle.dump(inputs,open("inputs_processed.pkl","wb"))

        if(os.path.isfile("targets_processed.pkl")):
            targets = cPickle.load(open("targets_processed.pkl","rb"))
        else:
            targets = [self.process_sentence(item, bot_input=True) for item in targets]
            targets = self.normalize_sentences_length(targets,targetSentences=True)
            cPickle.dump(targets,open("targets_processed.pkl","wb"))
        
        return inputs, targets

    def __create_vocab(self):
        bow = CountVectorizer()
        bow.fit(self.inputs.tolist() + self.targets.tolist())
        
        vocab = list(bow.vocabulary_.keys())
        vocab.insert(0,"_NUM")
        vocab.insert(0,"_UNK")
        vocab.insert(0,"<END>")
        vocab.insert(0,"<START>")
        vocab.insert(0,"<PAD>")
        cPickle.dump(vocab,open("vocab","wb"))

        return vocab

    def createDataset(self, dataSourcePath="./data/frames.json"):

        self.path = dataSourcePath
        self.stemmer = PorterStemmer()
        self.spell = Speller()
        self.input_doc, self.target_doc = self.__create_docs()
        # self.input_doc = self.input_doc[0:1000]
        # self.target_doc = self.target_doc[0:1000]
        self.inputs = np.array([message[1] for message in self.input_doc])
        self.input_lens = np.array([message[0] for message in self.input_doc])
        self.targets = np.array([message[1] for message in self.target_doc])
        self.target_lens = np.array([message[0] for message in self.target_doc])
        self.vocab = self.__create_vocab()

        print(self.targets[0])

        print(max(len(y.split(" ")) for y in self.targets))
        self.vocab_size = len(self.vocab) + 1

        # word2index
        self.vocab_features_dict = dict([(token, i) for i, token in enumerate(self.vocab)])
        # index2word
        self.reverse_features_dict = dict((i, token) for token, i in self.vocab_features_dict.items())

        max_encoder_seq_length = self.MAX_LENGTH
        max_decoder_seq_length = self.MAX_LENGTH+1

        encoder_input_data = np.full((len(self.inputs), max_encoder_seq_length), self.vocab_features_dict.get('<PAD>'), dtype='int32')
        decoder_input_data = np.full((len(self.inputs), max_decoder_seq_length), self.vocab_features_dict.get('<PAD>'), dtype='int32')

        print(decoder_input_data.shape)

        unknowId = self.vocab_features_dict.get('_UNK')

        print("# Create encoder input tensor")

        tokenized_sentences = [sentence.split(" ") for sentence in self.inputs]

        for i in range(len(self.input_lens)):
            ids_vect = [self.vocab_features_dict.get(tk, unknowId) for tk in tokenized_sentences[i][0:self.input_lens[i]]]
            encoder_input_data[i][0:len(ids_vect)] = ids_vect

        print("# Create decoder input tensor")

        tokenized_sentences = [sentence.split(" ") for sentence in self.targets]

        for i in range(len(self.target_lens)):
            ids_vect = [self.vocab_features_dict.get(tk, unknowId) for tk in tokenized_sentences[i][0:self.target_lens[i]]]
            decoder_input_data[i][0:len(ids_vect)] = ids_vect

        # Creating training and validation sets using an 80-20 split
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(encoder_input_data, decoder_input_data, test_size=0.2)
        
        self.BUFFER_SIZE = len(input_tensor_train)
        self.steps_per_epoch = self.BUFFER_SIZE

        print("# Create Dataset")

        self.dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
        example_X, example_Y = next(iter(self.dataset))
        




