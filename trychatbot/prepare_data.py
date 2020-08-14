
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer



from nltk.stem import PorterStemmer
from autocorrect import Speller

import os
from six.moves import cPickle
import re

# import tensorflow as tf
# import tensorflow_addons as tfa
# from tensorboard.plugins import projector

MAX_LEN = 25
BATCH_SIZE = 64
NUM_EPOCH = 10000

def normalize_data():
    data = pd.read_json("./data/frames.json")
    data['chat'] = data['turns'].apply(lambda x: [item['text'] for item in x])
    data['user'] = data['chat'].apply(lambda x: x[0::2])
    data['bot'] = data['chat'].apply(lambda x: x[1::2])
    dataset = data.apply(lambda x: list(itertools.zip_longest(x['user'], x['bot'], fillvalue='')), axis=1)
    all_convos = []
    _ = dataset.apply(lambda x: all_convos.extend(x))
    cPickle.dump(all_convos, open("all_convos.pkl","wb"))
   
stemmer = PorterStemmer()
spell = Speller()
def process_sentence(str, bot_input=False, bot_output=False):
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
    str = [stemmer.stem(re.sub(r'(.)\1+',r'\1\1',token)) for token in str]
    str = [spell(token).lower() for token in str]

    while True:
        try:
            str.remove("")
        except:
            break

    # if(not bot_input and not bot_output):
        # str = str[0:MAX_LEN]
    # elif(bot_input):
        # str = str[0:MAX_LEN-1]
    if(bot_input):
        str.insert(0,"<START>")
        str.insert(len(str),"<END>")
    # else:
    #     str = str[0:MAX_LEN-1]

    # old_len = len(str)
    # for i in range((MAX_LEN) - len(str)):
    #     str.append(" </pad> ")
    # str = re.sub("\s+"," "," ".join(str)).strip()
    return str #, old_len

# normalize_data()
# inputs = ["Hello World","Good Morning!"]
# x = [process_sentence(item) for item in inputs]
# print(x)
# x = np.array([message[0] for message in x])
# print(x[0])
# Load data

data = cPickle.load(open("all_convos.pkl","rb"))
inputs = [item[0] for item in data]
targets = [item[1] for item in data]

def append_btw(str,elem):
    str.insert(len(str)-1,elem)

def append(str, elem):
    str.append(elem)

def normalize_sentences_length(sentences, targetSentences=False):
    real_lens = [len(s) for s in sentences]
    bigger_sentence_size = max(real_lens)
    if (targetSentences):
        f = append_btw
    else:
        f = append
    for i in range(len(sentences)):
        str = sentences[i]
        for j in range((bigger_sentence_size) - len(str)):
            f(str," <PAD> ")
        sentences[i] = re.sub("\s+"," "," ".join(str)).strip()
    return list(zip(real_lens, sentences))

if(os.path.isfile("inputs_processed.pkl")):
    inputs = cPickle.load(open("inputs_processed.pkl","rb"))
else:
    inputs = [process_sentence(item) for item in inputs]
    normalize_sentences_length(inputs)
    cPickle.dump(inputs,open("inputs_processed.pkl","wb"))
if(os.path.isfile("targets_processed.pkl")):
    targets = cPickle.load(open("targets_processed.pkl","rb"))
else:
    targets = [process_sentence(item, bot_input=True) for item in targets]
    normalize_sentences_length(targets,targetSentences=True)
    cPickle.dump(targets,open("targets_processed.pkl","wb"))

# print(inputs)
# if(os.path.isfile("bot_out_processed.pkl")):
#     bot_outputs = cPickle.load(open("bot_out_processed.pkl","rb"))
# else:
#     bot_outputs = [process_sentence(item, bot_output=True) for item in bot]
#     cPickle.dump(user,open("bot_out_processed.pkl","wb"))

user = np.array([message[1] for message in inputs])
user_lens = np.array([message[0] for message in inputs]).astype(np.int32)
bot_inputs = np.array([message[1] for message in targets])
bot_in_lens = np.array([message[0] for message in targets]).astype(np.int32)
bot_outputs = np.array([message[1] for message in targets])
bot_out_lens = np.array([message[0] for message in targets]).astype(np.int32)

# bot_out_lens = np.array([message[1] for message in bot_outputs]).astype(np.int32)
# bot_outputs = np.array([message[0] for message in bot_outputs])
# print(inputs.tolist())
bow = CountVectorizer()

# print(user.tolist())
# exit()
bow.fit(user.tolist() + bot_inputs.tolist())

vocab = list(bow.vocabulary_.keys())
vocab.insert(0,"_NUM")
vocab.insert(0,"_UNK")
vocab.insert(0,"<END>")
vocab.insert(0,"<START>")
vocab.insert(0,"<PAD>")
cPickle.dump(vocab,open("vocab","wb"))

# user_ph = tf.compat.v1.placeholder(dtype=user.dtype, name="user_placeholder")
user_ph = tf.compat.v1.placeholder(dtype=user.dtype, name="user_placeholder")
bot_in_ph = tf.compat.v1.placeholder(dtype=bot_inputs.dtype, name="bot_in_placeholder")
bot_out_ph = tf.compat.v1.placeholder(dtype=bot_outputs.dtype, name="bot_out_placeholder")

user_lens_ph = tf.compat.v1.placeholder(dtype=user_lens.dtype, shape=[None], name="user_lens_placeholder")
bot_in_lens_ph = tf.compat.v1.placeholder(dtype=bot_in_lens.dtype, shape=[None], name="bot_in_lens_placeholder")
bot_out_lens_ph = tf.compat.v1.placeholder(dtype=bot_out_lens.dtype, shape=[None], name="bot_out_lens_placeholder")

tf_user = tf.data.Dataset.from_tensor_slices(user_ph)
tf_bot_in = tf.data.Dataset.from_tensor_slices(bot_in_ph)
tf_bot_out = tf.data.Dataset.from_tensor_slices(bot_out_ph)

tf_user_lens = tf.data.Dataset.from_tensor_slices(user_ph)
tf_bot_in_lens = tf.data.Dataset.from_tensor_slices(bot_in_lens_ph)
tf_bot_out_lens = tf.data.Dataset.from_tensor_slices(bot_out_lens_ph)

class Iterator:
    def __init__(self, dataset):
        self.dataset = dataset


with tf.device("/cpu:0"), tf.name_scope("data"):

    vocab_features_dict = dict([(token, i) for i, token in enumerate(vocab)])
    reverse_features_dict = dict((i, token) for token, i in vocab_features_dict.items())

    tf_user = tf_user.map(lambda string: tf.strings.split([string])).map(lambda token: vocab_features_dict.get(token, 3))
    tf_bot_in = tf_bot_in.map(lambda string: tf.strings.split([string])).map(lambda token: vocab_features_dict.get(token, 3))
    tf_bot_out = tf_bot_out.map(lambda string: tf.strings.split([string])).map(lambda token: vocab_features_dict.get(token, 3))

    data = tf.data.Dataset.zip((tf_user, tf_bot_in, tf_bot_out, tf_user_lens, tf_bot_in_lens, tf_bot_out_lens))
    data = data.shuffle(buffer_size=256).batch(BATCH_SIZE)
    data = data.prefetch(10)
    print(list(data.as_numpy_iterator()))
    exit()
    data_iterator = tf.compat.v1.data.Iterator.from_structure(
        tf.compat.v1.data.get_output_types(data), 
        tf.compat.v1.data.get_output_shapes(data), 
        None, 
        tf.compat.v1.data.get_output_classes(data)
        )
    train_init_op = data_iterator.make_initializer(data, name="dataset_init")
    user_doc, bot_in_doc, bot_out_doc, user_len, bot_in_len, bot_out_len = data_iterator.get_next()
    print(type(user_doc))
    user_doc = tf.sparse.to_dense(user_doc)
    bot_in_doc = tf.sparse.to_dense(bot_in_doc)
    bot_out_doc = tf.sparse.to_dense(bot_out_doc)

    print(user_doc)
    exit()

exit()




# Até aqui tem-se inputs,targets, e input_tokens/target_tokens (vocab)

# word2index
vocab_features_dict = dict([(token, i) for i, token in enumerate(vocab)])
# index2word
reverse_features_dict = dict((i, token) for token, i in vocab_features_dict.items())

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input)) for input in inputs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target)) for target in targets])

encoder_input_data = np.zeros((len(inputs), max_encoder_seq_length), dtype='int32')
decoder_input_data = np.zeros((len(inputs), max_decoder_seq_length), dtype='int32')
# decoder_target_data = np.zeros((len(inputs), max_decoder_seq_length, len(vocab)), dtype='float32')

# print(encoder_input_data)
# exit()

tokenized_sentences = [sentence.split(" ") for sentence in inputs]
tokenized_sentences = tokenized_sentences[0:5]
# print(encoder_input_data)

unknowId = vocab_features_dict.get('_UNK')

for encoder_input_data_line in encoder_input_data:
    for sentence_tks in tokenized_sentences:
        ids_vect = [vocab_features_dict.get(tk) if not vocab_features_dict.get(tk) is None else unknowId for tk in sentence_tks]
        encoder_input_data_line[0:len(ids_vect)] = ids_vect

tokenized_sentences = [sentence.split(" ") for sentence in targets]
tokenized_sentences = tokenized_sentences[0:5]

for decoder_input_data_line in decoder_input_data:
    for sentence_tks in tokenized_sentences:
        ids_vect = [vocab_features_dict.get(tk) if not vocab_features_dict.get(tk) is None else unknowId for tk in sentence_tks]
        decoder_input_data_line[0:len(ids_vect)] = ids_vect

# print(vocab_features_dict.get('is'))
# print(tokenized_sentences[0])
# tensor_input = [[vocab_features_dict.get(token) for token in ]]

print(encoder_input_data)
print(decoder_input_data)
exit()

# for line, (inputs, targets) in enumerate(zip(inputs, targets)):
#     for timestep, token in enumerate(re.findall(r"[\w']|[^\s\w]", inputs)):
#         encoder_input_data[line, timestep, vocab_features_dict[token]] = 1

#     for timestep, token in enumerate(targets.split()):
#         decoder_input_data[line, timestep, vocab_features_dict[token]] = 1
#         if timestep > 0:
#             decoder_target_data[line, timestep - 1, vocab_features_dict[token]] = 1

print(encoder_input_data)

exit()
#Dimensionality
dimensionality = 256
#The batch size and number of epochs
batch_size = 64
epochs = 1000
#Encoder
encoder_inputs = tf.keras.layers.Input(shape=(None, len(vocab)))
encoder_lstm = tf.keras.layers.LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]
#Decoder
decoder_inputs = tf.keras.layers.Input(shape=(None, len(vocab)))
decoder_lstm = tf.keras.layers.LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(len(vocab), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#Model
training_model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
#Compiling
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
#Training
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)
training_model.save('training_model.h5')

inputs_ph = tf.Tensor(dtype=inputs.dtype,name="inputs_placeholder")
targets_ph = tf.Tensor(dtype=targets.dtype,name="targets_placeholder")
# bot_in_ph = tf.compat.v1.placeholder(bot_inputs.dtype,name="bot_in_placeholders")
# bot_out_ph = tf.compat.v1.placeholder(bot_outputs.dtype,name="bot_out_placeholders")

# user_lens_ph = tf.compat.v1.placeholder(user_lens.dtype, shape=[None],name="user_lens_placeholder")
# bot_in_lens_ph = tf.compat.v1.placeholder(bot_in_lens.dtype, shape=[None],name="bot_in_lens_placeholder")
# bot_out_lens_ph = tf.compat.v1.placeholder(bot_out_lens.dtype, shape=[None],name="bot_out_lens_placeholder")

tf_inputs = tf.data.Dataset.from_tensor_slices(inputs_ph)
tf_bot_in = tf.data.Dataset.from_tensor_slices(targets_ph)
# tf_bot_out = tf.data.Dataset.from_tensor_slices(bot_out_ph)

# tf_user_lens = tf.data.Dataset.from_tensor_slices(user_lens_ph)
# tf_bot_in_lens = tf.data.Dataset.from_tensor_slices(bot_in_lens_ph)
# tf_bot_out_lens = tf.data.Dataset.from_tensor_slices(bot_out_lens_ph)

# with tf.device("/cpu:0"), tf.name_scope("data"):

#     init = tf.lookup.KeyValueTensorInitializer(tf.constant(vocab), tf.range(tf.size(vocab)))
#     words = tf.lookup.StaticHashTable(init,default_value=3)

#     tf_user = tf_user.map(lambda string: tf.strings.split([string])).map(lambda tokens: (words.lookup(tokens)))
#     tf_bot_in = tf_bot_in.map(lambda string: tf.strings.split([string])).map(lambda tokens: (words.lookup(tokens)))
#     tf_bot_out = tf_bot_out.map(lambda string: tf.strings.split([string])).map(lambda tokens: (words.lookup(tokens)))

#     data = tf.data.Dataset.zip((tf_user, tf_bot_in, tf_bot_out, tf_user_lens, tf_bot_in_lens, tf_bot_out_lens))
#     data = data.shuffle(buffer_size=256).batch(BATCH_SIZE)
#     data = data.prefetch(10)
#     data_iterator = tf.compat.v1.data.Iterator.from_structure(data.output_types, data.output_shapes, None, data.output_classes)
#     train_init_op = data_iterator.make_initializer(data, name="dataset_init")
#     user_doc, bot_in_doc, bot_out_doc, user_len, bot_in_len, bot_out_len = data_iterator.get_next()
#     user_doc = tf.sparse.to_dense(user_doc)
#     bot_in_doc = tf.sparse.to_dense(bot_in_doc)
#     bot_out_doc = tf.sparse.to_dense(bot_out_doc)

# with tf.name_scope("embedding"):
#     embedding = tf.compat.v1.get_variable("embedding", [len(vocab), 200], initializer=tf.compat.v1.glorot_uniform_initializer())

#     embedded_user = tf.compat.v1.nn.embedding_lookup(embedding, user_doc)
#     embedded_user_dropout = tf.compat.v1.nn.dropout(embedded_user, 0.7)

#     embedded_bot_in = tf.compat.v1.nn.embedding_lookup(embedding, bot_in_doc)
#     embedded_bot_in_dropout = tf.compat.v1.nn.dropout(embedded_bot_in, 0.7)

#     embedded_user_dropout = tf.compat.v1.reshape(embedded_user_dropout, [-1, MAX_LEN, 200])
#     embedded_bot_in_dropout = tf.compat.v1.reshape(embedded_bot_in_dropout, [-1, MAX_LEN, 200])

# with tf.name_scope("encoder"):
#     encoder_GRU = tf.compat.v1.nn.rnn_cell.GRUCell(128)
#     encoder_cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(encoder_GRU,  input_keep_prob=0.7, 
#                     output_keep_prob=0.7, state_keep_prob=0.9)
#     encoder_cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(encoder_GRU,  input_keep_prob=0.7, 
#                     output_keep_prob=0.7, state_keep_prob=0.9)
    
#     encoder_outputs, encoder_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(
#         encoder_cell_fw, encoder_cell_bw, embedded_user_dropout, sequence_length=user_len, dtype=tf.float32)

#     encoder_state = tf.concat(encoder_state, 1)

# with tf.name_scope("projection"):
#     projection_layer = tf.compat.v1.layers.Dense(len(vocab), use_bias=False)

# with tf.name_scope("decoder"):
#     decoder_GRU = tf.compat.v1.nn.rnn_cell.GRUCell(256)
#     decoder_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(decoder_GRU, input_keep_prob=0.7, 
#                     output_keep_prob=0.7, state_keep_prob=0.9)

#     # Helper for use during training
#     # During training we feed the decoder
#     # the target sequence
#     # However, during testing we use the decoder's
#     # last output
#     helper = tfa.seq2seq.TrainingSampler(embedded_bot_in_dropout, bot_in_len)

#     decoder = tfa.seq2seq.BasicDecoder(
#         decoder_cell, helper, encoder_state, output_layer=projection_layer
#     )

#     outputs, _, _ = tfa.seq2seq.dynamic_decode(decoder)

#     logits = outputs.rnn_output

#     translations = outputs.sample_id

# with tf.name_scope("loss"):
#     loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(bot_out_doc, [-1, MAX_LEN]), logits=logits)
#     mask = tf.sequence_mask(bot_out_len, dtype=tf.float32)
#     train_loss = (tf.reduce_sum(loss * mask) / BATCH_SIZE)

# with tf.compat.v1.variable_scope('Adam'):
#     global_step = tf.Variable(0, trainable=False)
#     inc_gstep = tf.compat.v1.assign(global_step, global_step + 1)
#     learning_rate = tf.compat.v1.train.cosine_decay_restarts(0.001, global_step, 550, t_mul=1.1)
#     adam_optmizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
#     adam_gradients, v = zip(*adam_optmizer.compute_gradients(train_loss))
#     adam_gradients, _ = tf.clip_by_global_norm(adam_gradients, 10.0)
#     adam_optmize = adam_optmizer.apply_gradients(zip(adam_gradients, v))

# with tf.compat.v1.variable_scope("inference"):
#     # Helper
#     # Start token is 1, which is the </start> token
#     # End token is 2
#     helper = tfa.seq2seq.GreedyEmbeddingSampler(
#         embedding, tf.fill([BATCH_SIZE], 1), 2
#     )

#     # Decoder
#     decoder = tfa.seq2seq.BasicDecoder(
#         decoder_cell, helper, encoder_state, output_layer=projection_layer
#     )

#     # Dynamic decoding
#     test_outputs, _, _ = tfa.seq2seq.dynamic_decode(
#         decoder, maximum_iterations=10
#     )

#     test_translations = tf.identity(test_outputs.sample_id, name="word_ids")
#     test_words = tf.identity(inverse.lookup(tf.cast(test_translations, tf.int64)), name="words")

# with tf.name_scope("summaries"):
#     tf.summary.scalar('Loss', train_loss)
#     tf.summary.scalar('LR', learning_rate)
#     merged = tf.summary.merge_all()
#     config = projector.ProjectorConfig()
#     embedding_vis = config.embeddings.add()
#     embedding_vis.tensor_name = embedding.name
#     vocab_str = '\n'.join(vocab)
#     metadata = pd.Series(vocab)
#     metadate.name = "label"
#     metadata.to_csv("checkpoints/metadata.tsv", sep="\t", header=True, index_label="index")
#     embedding_vis.metadata_path = 'metadata.tsv'

# losses = []
# print("Started Training")
# saver = tf.compat.v1.train.Saver()
# save_dir = 'checkpoints/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_path = os.path.join(save_dir, 'best_validation')
# sess = tf.compat.v1.InteractiveSession(sconfig)
# writer = tf.compat.v1.summary.FileWriter('./checkpoints', sess.graph)
# projector.visualize_embeddings(writer, config)
# sess.run([words.init, tf.compat.v1.global_variables_initializer(), inverse.init])

# step = 0
# for i in range(NUM_EPOCH):
#     if(i % 10 == 0):
#         saver.save(sess=sess, save_path=save_path, write_meta_graph=True)
#     sess.run(train_init_op, feed_dict={
#         user_ph: user,
#         bot_in_ph: bot_inputs,
#         bot_out_ph: bot_outputs,
#         user_lens_ph: user_lens,
#         bot_in_lens_ph: bot_in_lens,
#         bot_out_lens_ph: bot_out_lens
#     })

#     while True:
#         try:
#             _, batch_loss, summary = sess.run([adam_optmize, train_loss, merged])
#             writer.add_summary(summary, i)
#             losses.append(batch_loss)
#         except tf.errors.InvalidArgumentError:
#             continue
#         except tf.errors.OutOfRangeError:
#             print("Epoch {}: Loss(Mean): {} Loss(Std): {}".format(i, np.mean(losses), np.std(losses)))
#             losses = []
#             break
#         sess.run(inc_gstep)
#         step += 1

# if __name__ == "__main__":
#     str,len = process_sentence("There's a man ?", bot_input=True)
#     print(str)