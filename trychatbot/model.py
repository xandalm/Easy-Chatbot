
import time
import os
import re
import numpy as np
import random

import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa

from data.prepare_data import Data

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(enc_units,
                                    return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, [state_h, state_c]

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                    return_sequences=True,
                                    return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size, activation='softmax')

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        print(context_vector.shape)
        print(x.shape)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state_h, state_c = self.lstm(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, [state_h, state_c], attention_weights

    def build_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

#ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )
    
#DECODER
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units, dense_units, batch_size, tx):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,batch_size*[tx])
        self.rnn_cell =  self.build_rnn_cell(batch_size, dense_units)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units, memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        #return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size, dense_units):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell
    
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state


class Model:
    def __init__(self, embedding_dims=256, rnn_units=1024, dense_units=1024):
        self.save_ckpt = False
        self.embedding_dims = embedding_dims
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        # try:
            # restoring the latest checkpoint in checkpoint_dir
            # print("Success in load model from checkpoint")
        # except:
            # raise Exception("Failed to load checkpoint")
        # exit()

    def load(self, data, with_checkpoint=False):
        self.data = data

        # 
        # 
        # self.enc_model, self.dec_model = self.__make_inference_models()
        # 
        # 
        # ?

        # # With TensorFlow Addons

        self.encoder = EncoderNetwork(self.data.vocab_size, self.embedding_dims, self.rnn_units)
        self.decoder = DecoderNetwork(self.data.vocab_size, self.embedding_dims, self.rnn_units, 
                                            self.dense_units, self.data.BATCH_SIZE, self.data.Tx)

        # # No TensorFlow Addons

        # self.encoder = Encoder(self.data.vocab_size, self.embedding_dims, self.rnn_units, self.data.BATCH_SIZE)
        # self.decoder = Decoder(self.data.vocab_size, self.embedding_dims, self.rnn_units, self.data.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam()

        self.checkpoint_dir = './training_checkpoints/'+data.__class__.__name__
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                        encoder=self.encoder,
                                        decoder=self.decoder)
        if with_checkpoint and os.path.exists(self.checkpoint_dir+"/checkpoint"):
            print('Loaded')
            self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir)).expect_partial()

    def __loss_function(self, y_pred, y):
        #shape of y [batch_size, ty]
        #shape of y_pred [batch_size, Ty, output_vocab_size] 
        sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                      reduction='none')
        loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
        mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
        mask = tf.cast(mask, loss.dtype)
        loss = mask* loss
        loss = tf.reduce_mean(loss)
        return loss

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train_step(self, inp, targ):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp)

            dec_hidden = enc_hidden
            
            dec_input = tf.expand_dims([self.data.vocab_features_dict.get('<START>')] * self.data.BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def __step(self, input_batch, output_batch,encoder_initial_cell_state):
        loss = 0 # initialize loss

        with tf.GradientTape() as tape:
            encoder_emb_inp = self.encoder.encoder_embedding(input_batch)
            a, a_tx, c_tx = self.encoder.encoder_rnnlayer(encoder_emb_inp, 
                                                            initial_state =encoder_initial_cell_state)

            #[last step activations,last memory_state] of encoder passed as input to decoder Network
            
            # Prepare correct Decoder input & output sequence data
            decoder_input = output_batch[:,:-1] # ignore <end>
            #compare logits with timestepped +1 version of decoder_input
            decoder_output = output_batch[:,1:] #ignore <start>

            # Decoder Embeddings
            decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

            #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
            self.decoder.attention_mechanism.setup_memory(a)
            decoder_initial_state = self.decoder.build_decoder_initial_state(self.data.BATCH_SIZE,
                                                                              encoder_state=[a_tx, c_tx],
                                                                              Dtype=tf.float32)
            #BasicDecoderOutput        
            outputs, _, _ = self.decoder.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                                  sequence_length=self.data.BATCH_SIZE*[self.data.Ty-1])

            logits = outputs.rnn_output
            #Calculate loss

            loss = self.__loss_function(logits, decoder_output)

        #Returns the list of all layer variables / weights.
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables  
        # differentiate loss wrt variables
        gradients = tape.gradient(loss, variables)

        #grads_and_vars – List of(gradient, variable) pairs.
        grads_and_vars = zip(gradients,variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss
    
    def __initialize_initial_state(self):
        return [tf.zeros((self.data.BATCH_SIZE, self.rnn_units)), tf.zeros((self.data.BATCH_SIZE, self.rnn_units))]

    def setEpochs(self, newVal):
        self.EPOCHS = newVal

    def queueCkptSave(self):
        print("Enqueue to save checkpoint this Epoch. Wait... ")
        self.save_ckpt = True
    
    def train(self, epochs=10000):
        if self.data is None:
            raise Exception("Haven't data to train")

        print("# Init. Train in {} epochs... ".format(epochs))

        self.EPOCHS = epochs
        i = 0
        while i < self.EPOCHS:
            start = time.time()
            encoder_initial_cell_state = self.__initialize_initial_state()
            total_loss = 0.0
            i+=1
            for ( batch, (input_batch, output_batch)) in enumerate(self.data.dataset.take(self.data.steps_per_epoch)):
                batch_loss = self.__step(input_batch, output_batch, encoder_initial_cell_state)
                total_loss += batch_loss
                if batch % 20 == 0:
                    print("Epoch {} Batch {} Loss {:.4}".format(i, batch, batch_loss.numpy()))

            if i % 1000 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print("Epoch {} Loss {:.4}".format(i,total_loss/self.data.steps_per_epoch))
            print("Time taken for 1 epoch {} sec\n".format(time.time() - start))
    
    def train_2(self, epochs=1000):
        print("# Init. Train in {} epochs... ".format(epochs))

        self.EPOCHS = epochs

        for epoch in range(self.EPOCHS):
            start = time.time()

            # enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.data.dataset.take(self.data.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ)
                total_loss += batch_loss

                if batch % 20 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / self.data.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def __evaluate(self, sentence):
        import re
        sentence = self.data.process_sentence(sentence)
        input_lines = ["<START> "+sentence]
        input_sequences = [[self.data.vocab_features_dict.get(w, 3) for w in line.split(" ")] for line in input_lines]

        input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                                maxlen=self.data.Tx, padding='post')
        input = tf.convert_to_tensor(input_sequences)
        inference_batch_size = input_sequences.shape[0]

        encoder_initial_cell_state = [tf.zeros((inference_batch_size, self.rnn_units)),
                                    tf.zeros((inference_batch_size, self.rnn_units))]
        encoder_emb_inp = self.encoder.encoder_embedding(input)
        a, a_tx, c_tx = self.encoder.encoder_rnnlayer(encoder_emb_inp,
                                                        initial_state =encoder_initial_cell_state)

        start_tokens = tf.fill([inference_batch_size],self.data.vocab_features_dict.get("<START>"))

        end_token = self.data.vocab_features_dict.get("<END>")

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_input = tf.expand_dims([self.data.vocab_features_dict.get("<START>")]* inference_batch_size,1)
        decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

        decoder_instance = tfa.seq2seq.BasicDecoder(cell = self.decoder.rnn_cell, sampler = greedy_sampler,
                                                    output_layer=self.decoder.dense_layer)
        self.decoder.attention_mechanism.setup_memory(a)
        #pass [ last step activations , encoder memory_state ] as input to decoder for LSTM
        decoder_initial_state = self.decoder.build_decoder_initial_state(inference_batch_size,
                                                                        encoder_state=[a_tx, c_tx],
                                                                        Dtype=tf.float32)

        # Since we do not know the target sequence lengths in advance, we use maximum_iterations to limit the translation lengths.
        # One heuristic is to decode up to two times the source sentence lengths.
        maximum_iterations = tf.round(tf.reduce_max(self.data.Tx) * 2)

        #initialize inference decoder
        decoder_embedding_matrix = self.decoder.decoder_embedding.variables[0] 
        (first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                    start_tokens = start_tokens,
                                    end_token=end_token,
                                    initial_state = decoder_initial_state)

        inputs = first_inputs
        state = first_state  
        predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             
        for j in range(maximum_iterations):
            outputs, next_state, next_inputs, finished = decoder_instance.step(j,inputs,state)
            inputs = next_inputs
            state = next_state
            outputs = np.expand_dims(outputs.sample_id,axis = -1)
            predictions = np.append(predictions, outputs, axis = -1)

        for i in range(len(predictions)):
            line = predictions[i,:]
            seq = list(itertools.takewhile( lambda index: index !=2, line))
            try:
                seq = seq[:seq.index(self.data.vocab_features_dict.get("<PAD>"))]
                seq = seq[:seq.index(self.data.vocab_features_dict.get("<END>"))]
            except:
                pass
            return " ".join( [self.data.reverse_features_dict.get(w,"_UNK") for w in seq])

    def __evaluate_2(self, sentence):
        attention_plot = np.zeros((self.data.Ty, self.data.Tx))

        # sentence = preprocess_sentence(sentence)

        inputs = [self.data.vocab_features_dict.get(k, 3) for k in sentence.split(" ")]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.data.Tx,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.rnn_units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.data.vocab_features_dict.get('<START>')], 0)

        for t in range(self.data.Ty):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.data.reverse_features_dict.get(predicted_id) + ' '

            if self.data.reverse_features_dict.get(predicted_id) == '<end>':
                return result#, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result#, sentence, attention_plot

    def get_response(self, message):
        message = message.strip()
        END_CHAT=["Tchau", "Bye", "Até mais", "Até", "Ate mais", "Ate"]
        _ = False
        if len(message) < 10:
            aux = message.lower()
            for y in END_CHAT:
                if aux == y.lower():
                    _ = True
                    break
        if _:
            return (False, random.choice(END_CHAT))
        if message == "":
            return (True, random.choice(["O que foi?","Diga algo"]))
        with tf.device("/cpu:0"):
            result = self.__evaluate(message)

        return (True, result[0].upper()+result[1:])


    def __make_model(self):
        self.encoder_inputs = tf.keras.layers.Input(shape=(None,))
        self.encoder_embedding = tf.keras.layers.Embedding(self.data.vocab_size, self.embedding_dims, mask_zero=True)(self.encoder_inputs)
        enconder_outputs, state_h, state_c = tf.keras.layers.LSTM(self.embedding_dims,
                                return_state=True)(self.encoder_embedding)
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = tf.keras.layers.Input(shape=(None,))
        self.decoder_embedding = tf.keras.layers.Embedding(self.data.vocab_size, self.embedding_dims, mask_zero=True)(self.decoder_inputs)
        self.decoder_lstm = tf.keras.layers.LSTM(self.embedding_dims, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_embedding, initial_state=encoder_states)
        self.decoder_dense = tf.keras.layers.Dense(self.data.vocab_size, activation='softmax')
        self.output = self.decoder_dense(decoder_outputs)

    def train_3(self, epochs=100):
        # self.__make_model()
        encoder_inputs = tf.keras.layers.Input(shape=(None,))
        encoder_embedding = tf.keras.layers.Embedding(self.data.num_encoder_tokens, self.embedding_dims, mask_zero=True)(encoder_inputs)
        enconder_outputs, state_h, state_c = tf.keras.layers.LSTM(self.embedding_dims,
                                return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = tf.keras.layers.Input(shape=(None,))
        decoder_embedding = tf.keras.layers.Embedding(self.data.num_decoder_tokens, self.embedding_dims, mask_zero=True)(decoder_inputs)
        decoder_lstm = tf.keras.layers.LSTM(self.embedding_dims, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(self.data.num_decoder_tokens, activation='softmax')
        output = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)

        # Compile & run training
        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!
        model.summary()
        model.fit([self.data.encoder_input_data, self.data.decoder_input_data], self.data.decoder_target_data,
                batch_size=self.data.BATCH_SIZE,
                epochs=epochs,
                validation_split=0.2, verbose=2)
                
        model.save(self.data.path + 'best_bot_version2.h5')        

        model.save_weights(self.data.path + "model_bot_version2.h5")


    def __make_inference_models(self):
        model = tf.keras.models.load_model(self.data.path + "best_bot_version2.h5")
        model.load_weights(self.data.path + "model_bot_version2.h5")
        # print(model.input)
        # exit()
        encoder_inputs = model.input[0]
        encoder_outputs, state_h, state_c = model.layers[4].output
        encoder_states = [state_h, state_c]
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
        

        decoder_inputs = model.input[1]

        decoder_state_input_h = tf.keras.layers.Input(shape=(self.embedding_dims ,), name="input_3")
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.embedding_dims ,), name="input_4")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        
        decoder_embedding = model.layers[3](decoder_inputs)
        decoder_lstm = model.layers[5]#tf.keras.layers.LSTM(self.embedding_dims, return_sequences=True, return_state=True)
        # print(decoder_lstm)
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding , initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_dense = model.layers[6]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = tf.keras.models.Model(
            [decoder_inputs] + decoder_states_inputs, 
            [decoder_outputs] + decoder_states)
        
        return encoder_model , decoder_model

    def __str_to_tokens(self, sentence):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append(self.data.vocab_features_dict.get(word, self.data.vocab_features_dict.get('_UNK')) )
        return tokens_list + [0] * (self.data.Tx - len(tokens_list))

    def __evaluate_3(self, sentence):
        with tf.device("/cpu:0"):
            states_values = self.enc_model.predict(self.__str_to_tokens(sentence))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self.data.vocab_features_dict.get("<START>")
        stop_condition = False
        decoded_translation = ''
        # states_values = empty_target_seq + states_values
        # for i, l in enumerate(states_values):
        #     print("index i: {}".format(i))
        #     for j, y in enumerate(l):
        #         print("index --> {}".format(j))
        #         print(y)
        # print(np.array([empty_target_seq.shape] + states_values).shape)
        # # print(len(states_values))
        # # print(len(states_values[0]))
        # # print(len(states_values[0][0]))
        # # exit()
        # exit()
        while not stop_condition:
            if sentence == 'Goodbye':
                stop_condition = True
            dec_outputs, h, c = self.dec_model.predict([empty_target_seq] + states_values )
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in self.data.reverse_features_dict.items():
                #print(sampled_word)
                if sampled_word_index == index:
                    if word != '<END>':
                        decoded_translation += ' {}'.format(word)
                    sampled_word = word
            
            if sampled_word == '<END>' or len(decoded_translation.split()) > self.data.Ty:
                stop_condition = True
            
                
            empty_target_seq = np.zeros((1 , 1))  
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]
        return decoded_translation