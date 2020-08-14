import numpy as np
import tensorflow as tf
import re
import time
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import layers , activations , models , preprocessing , utils
import random
import nltk
import itertools
from collections import defaultdict
import numpy as np
import pickle

# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
# sess = tf.Session(config=config) 
# sess = tf.Sess
# tf.keras.backend.set_session(sess)
# tf.keras.backend.set

import requests, zipfile, io

r = requests.get('http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip') 
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~\''

limit = {
        'maxq' : 10,
        'minq' : 2,
        'maxa' : 10,
        'mina' : 2
        }

UNK = 'unk'
VOCAB_SIZE = 5000

'''
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''
def get_id2line():
    lines=open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line

'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''
def get_conversations():
    conv_lines = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
    convs = [ ]
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
        convs.append(_line.split(','))
    return convs

'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''
def extract_conversations(convs,id2line,path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx)+'.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1

'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''
def gather_dataset(convs, id2line):
    questions = []; answers = []

    for conv in convs:
        if len(conv) %2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i%2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers

'''
 remove anything that isn't in the vocabulary
    return str(pure en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])

'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    # filt_data_len = len(filtered_q)
    # filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    # print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [x[0] for x in vocab]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences

'''
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

def process_data():

    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')
    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs,id2line)

    # change to lower case (just for en)
    # Add sos and eos 
    questions = [ line.lower() for line in questions ]
    answers = [ '<sos> ' + line.lower() + ' <eos>'for line in answers ]

    # filter out unnecessary characters
    print('\n>> Filter lines')
    questions = [ filter_line(line, EN_WHITELIST) for line in questions ]
    answers = [ filter_line(line, EN_WHITELIST) for line in answers ]

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(questions, answers)

    for q,a in zip(qlines[141:145], alines[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines ]
    atokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in alines ]
    print('\n:: Sample from segmented list of words')

    for q,a in zip(qtokenized[141:145], atokenized[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # indexing -> idx2w, w2idx
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    print('\n >> Filter Unknowns')
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
    print('\n Final dataset len : ' + str(len(qtokenized)))


    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count/word_count)))
    print('Dataset count : ' + str(idx_q.shape[0]))


process_data()


def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

metadata, idx_q, idx_a = load_data()

word_dict = metadata['w2idx']

index_dict = metadata['idx2w']

MAX_NB_WORDS = 100
MAX_SEQUENCE_LENGTH = 100

from collections import deque

def target_answer(item):
    item = deque(item)
    item.rotate(-1)  
    item[-1] = 0
    return item

decoder_target_data =  np.array([target_answer(i) for i in idx_a])
encoder_input_data = idx_q
decoder_input_data = idx_a

num_encoder_tokens = max([max(i) for i in encoder_input_data])
num_decoder_tokens = max([max(i) for i in decoder_input_data])

# Model Hiperparameters

num_encoder_tokens = max([max(i) for i in encoder_input_data]) + 1
num_decoder_tokens = max([max(i) for i in decoder_input_data]) + 1
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.

# Model Construction

# Define an input sequence and process it.
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
enconder_outputs, state_h, state_c = tf.keras.layers.LSTM(latent_dim,
                           return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
output = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)

# Compile & run training
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, verbose=2)

# Inference Model

max_question_len = encoder_input_data.shape[1]
max_answer_len = decoder_input_data.shape[1]

def make_inference_models():    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model

def str_to_tokens(sentence):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(word_dict.get(word, word_dict['unk']) )
    return tokens_list + [0] * (max_question_len - len(tokens_list))

# Talking with Bot

enc_model , dec_model = make_inference_models()

for _ in range(10):
    question = input( 'You: ' )
    states_values = enc_model.predict(str_to_tokens(question))
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = word_dict['sos']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        if question == 'Goodbye':
            stop_condition = True
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values )
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in word_dict.items():
            #print(sampled_word)
            if sampled_word_index == index:
                if word != 'eos':
                    decoded_translation += ' {}'.format(word)
                sampled_word = word
        
        if sampled_word == 'eos' or len(decoded_translation.split()) > max_answer_len:
            stop_condition = True
        
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c] 
    print('KerasBot: ' + decoded_translation)

model.save('best_bot_version2.h5')        

model.save_weights("model_bot_version2.h5")


