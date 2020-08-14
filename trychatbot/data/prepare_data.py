
from six.moves import cPickle
from autocorrect import Speller
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

import spacy

import numpy as np
import nltk
import itertools

import json
import re
import os
import tensorflow as tf

class Data(ABC):
    

    def __init__(self):
        self.padToken = {'word': '<PAD>','idx': 0}
        self.startToken = {'word': '<START>','idx': 1}
        self.endToken = {'word': '<END>','idx': 2}
        self.unkToken = {'word': '_UNK','idx': 3}
        self.MAX_LENGTH = 16
        self.BATCH_SIZE = 64
        self.__VOCAB_SIZE = 5000
        # self.Dtype = tf.float32

        self.LIMIT = {
            'maxq' : 30,
            'minq' : 1,
            'maxa' : 30,
            'mina' : 1
            }
        self.nlp = spacy.load("pt_core_news_sm")

    @abstractmethod
    def _normalize_data(self):
        pass

    def loadVocab(self):
        try:
            self.vocab = cPickle.load(open(self.path + "vocab","rb"))
            self.vocab_size = len(self.vocab)

            config = json.load(open(self.path + "data_config.json","r"))
            
            self.Tx = self.MAX_LENGTH_INPUT = config['max_length_input']
            self.Ty = self.MAX_LENGTH_TARGET = config['max_length_target']

            # word2index
            self.vocab_features_dict = dict([(token, i) for i, token in enumerate(self.vocab)])
            # index2word
            self.reverse_features_dict = dict((i, token) for token, i in self.vocab_features_dict.items())
        except:
            self.vocab = []

    def process_sentence_BR(self, str, bot_input=False):
        tokens = self.nlp(str)
        str = [t.text.lower() for t in tokens]
        if(bot_input):
            str = ["<START>"]+str[:self.MAX_LENGTH-1]+["<END>"]
        else:
            str = str[:self.MAX_LENGTH]
        return str

    def filter(self, inputs, targets):
        f_i, f_t = [], []
        data_len = len(inputs)

        assert data_len == len(targets)

        for i in range(data_len):
            ilen, tlen = len(inputs[i].split(' ')), len(targets[i].split(' '))
            if ilen >= self.LIMIT['minq'] and ilen <= self.LIMIT['maxq']:
                if tlen >= self.LIMIT['mina'] and tlen <= self.LIMIT['maxa']:
                    f_i.append(inputs[i])
                    f_t.append(targets[i])

        return f_i, f_t

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
        str = [re.sub(r"[0-9]+","_NUM",token) for token in str]
        # str = [self.stemmer.stem(re.sub(r'(.)\1+',r'\1\1',token)) for token in str]
        # str = [self.spell(token).lower() for token in str]

        while True:
            try:
                str.remove("")
            except:
                break
        if(bot_input):
            # str = str[:self.MAX_LENGTH-1]
            str.insert(0,"<START>")
            str.append("<END>")
        # else:
            # str = str[:self.MAX_LENGTH]
        str = re.sub("\s+"," "," ".join(str)).strip()
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
    
    def __create_vocabulary(self):

        freq_dist = nltk.FreqDist(itertools.chain(*self.inputs + self.targets))
        
        vocab = [v[0] for v in freq_dist.most_common(self.__VOCAB_SIZE)]

        vocab.insert(0,"_UNK")
        vocab.insert(0,"<PAD>")

        cPickle.dump(vocab,open(self.path + "vocab","wb"))
    
    def __create_vocab(self):

        # get frequency distribution
        freq_dist = nltk.FreqDist(itertools.chain(*self.inputs + self.targets))
        # get vocabulary of 'vocab_size' most used words
        vocab = [v[0] for v in freq_dist.most_common(self.__VOCAB_SIZE)]
        vocab.remove("<START>")
        vocab.remove("<END>")
        # index2word
        # index2word = ['<PAD>','<START>','<END>','_UNK'] + [x[0] for x in vocab]
        # word2index
        # word2index = dict([(w,i) for i,w in enumerate(index2word)] )

        # vocab = {
        #     'w2idx': word2index,
        #     'idx2w': index2word
        # }


        # from collections import defaultdict
        # frequency = defaultdict(int)
        # all_lines = 
        # for line in all_lines:
        #     for token in line:
        #         frequency[token] += 1
        # frequency.pop("<PAD>")
        # frequency.pop("<START>")
        # frequency.pop("<END>")
        # frequency = {t: f for t, f in frequency.items() if f > 1}
        # frequency = {t: f for t, f in sorted(frequency.items(), key=lambda item: item[1], reverse=True)}
        # vocab = [t for t in frequency.keys()]
        # # bow = CountVectorizer()
        # # bow.fit(self.inputs.tolist() + self.targets.tolist())
        
        # # vocab = list(bow.vocabulary_.keys())
        vocab.insert(0,"_UNK")
        vocab.insert(0,"<END>")
        vocab.insert(0,"<START>")
        vocab.insert(0,"<PAD>")

        cPickle.dump(vocab,open(self.path + "vocab","wb"))

        # return vocab
    
    def __create_docs(self):
        if(not os.path.isfile(self.path + "all_convos.pkl")):
            self._normalize_data()

        data = cPickle.load(open(self.path + "all_convos.pkl","rb"))
        inputs = [item[0] for item in data]
        targets = [item[1] for item in data]

        if(os.path.isfile(self.path + "inputs_processed.pkl")):
            inputs = cPickle.load(open(self.path + "inputs_processed.pkl","rb"))
        else:
            print("init create docs")
            inputs = [self.process_sentence(item) for item in inputs]
            # inputs = self.normalize_sentences_length(inputs)
            cPickle.dump(inputs,open(self.path + "inputs_processed.pkl","wb"))

        if(os.path.isfile(self.path + "targets_processed.pkl")):
            targets = cPickle.load(open(self.path + "targets_processed.pkl","rb"))
        else:
            targets = [self.process_sentence(item, bot_input=True) for item in targets]
            # targets = self.normalize_sentences_length(targets,targetSentences=True)
            cPickle.dump(targets,open(self.path + "targets_processed.pkl","wb"))
        inputs, targets = self.filter(inputs, targets)
        return inputs, targets

    def __makeTensor(self, tensor, m, unk):
        for i in range(len(m)):
            ids_vect = [self.vocab_features_dict.get(tk, unk) for tk in m[i]]
            tensor[i][0:len(ids_vect)] = ids_vect
        return tensor

    def createDataset(self):

        # self.stemmer = PorterStemmer()
        self.spell = Speller()
        self.input_doc, self.target_doc = self.__create_docs()
        self.input_doc = self.input_doc[0:self.dataset_size]
        self.target_doc = self.target_doc[0:self.dataset_size]
        

        # print(len(self.input_doc))
        # print()
        # print(len(self.target_doc))
        # self.inputs = np.array([message for message in self.input_doc])
        # self.input_lens = np.array([message for message in self.input_doc])
        # self.targets = np.array([message for message in self.target_doc])
        # self.target_lens = np.array([message for message in self.target_doc])

        self.inputs = [sentence.split(" ") for sentence in self.input_doc]
        self.targets = [sentence.split(" ") for sentence in self.target_doc]

        # exit()

        self.__create_vocab()
        self.vocab = cPickle.load(open(self.path + "vocab","rb"))
        self.vocab_size = len(self.vocab)

        max_encoder_seq_length = self.Tx = self.MAX_LENGTH_INPUT = max(len(l) for l in self.inputs)
        max_decoder_seq_length = self.Ty = self.MAX_LENGTH_TARGET = max(len(l) for l in self.targets)

        print("Vocab_size: {}".format(self.vocab_size))
        print("Max_enc_len: {}".format(self.Tx))
        print("Max_dec_len: {}".format(self.Ty))
        print("Data_size: {}".format(len(self.inputs)))
        # exit()
        json.dump({
            'vocab_size': self.vocab_size,
            'max_length_input': self.MAX_LENGTH_INPUT,
            'max_length_target': self.MAX_LENGTH_TARGET
        }, open(self.path + "data_config.json","w"))

        # word2index
        self.vocab_features_dict = dict([(token, i) for i, token in enumerate(self.vocab)])

        # index2word
        self.reverse_features_dict = dict((i, token) for token, i in self.vocab_features_dict.items())

        unknowId = self.vocab_features_dict.get('_UNK')

        # try:
        #     self.encoder_input_data = np.load(self.path + 'input_tensor.npy')
        #     self.decoder_input_data = np.load(self.path + 'target_tensor.npy')
        #     print("Loaded pre-built Tensors")
        # except:
        print("No Tensors found.")
        self.encoder_input_data = np.full((len(self.inputs), max_encoder_seq_length), self.vocab_features_dict.get('<PAD>'), dtype='int32')
        self.decoder_input_data = np.full((len(self.inputs), max_decoder_seq_length), self.vocab_features_dict.get('<PAD>'), dtype='int32')
        print("# Create encoder input tensor")

        self.encoder_input_data = self.__makeTensor(self.encoder_input_data, self.inputs, unknowId)

        print("# Create decoder input tensor")

        self.decoder_input_data = self.__makeTensor(self.decoder_input_data, self.targets, unknowId)

            # Creating training and validation sets using an 80-20 split
            # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(encoder_input_data, decoder_input_data, test_size=0.0)

            # Save Tensors

            # np.save(self.path + 'input_tensor.npy',self.encoder_input_data)
            # np.save(self.path + 'target_tensor.npy',self.decoder_input_data)

        self.num_encoder_tokens = self.vocab_size
        self.num_decoder_tokens = self.vocab_size


        from collections import deque

        def target_answer(item):
            item = deque(item)
            item.rotate(-1)  
            item[-1] = 0
            return item

        self.decoder_target_data =  np.array([target_answer(i) for i in self.decoder_input_data])
        self.BUFFER_SIZE = len(self.encoder_input_data)
        self.steps_per_epoch = self.BUFFER_SIZE

        print("# Create Dataset")

        self.dataset = tf.data.Dataset.from_tensor_slices(
                (self.encoder_input_data, self.decoder_input_data)
        ).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

        
        example_X, example_Y = next(iter(self.dataset))
        