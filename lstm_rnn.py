#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:49:38 2018

@author: Amy Bryce, Theresa Pekarek-Rosin
https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
https://github.com/enriqueav/lstm_lyrics

Inspiration for next steps:
https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3
"""

import numpy as np
#import pandas as pd
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional, Embedding
#from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import time
#import random
#import sys
#import io


#constants 
MIN_WORD_FREQUENCY = 1
SEQUENCE_LEN = 10
STEP = 1
BATCH_SIZE = 32

# split the data into test and train data -> default: 98/2 -> for small sample 90/10
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=10):
    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    #print("Size of training set = %d" % len(x_train))
    #print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)

# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y

def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024)) #turns indexes into dense vectors of fixed size
    model.add(Bidirectional(LSTM(128)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()


if __name__ == "__main__":
    # preprocessing
    #TODO: work out a way to circumvent the hyphenation problem 
    text_raw = (open("sample.txt").read())
    text_raw = text_raw.lower()
    text_raw = text_raw.replace('\n', ' \n ')
    #print ('length: ', len(sample_raw))
    
    # turns text in array containing the corpus word for word
    text_in_words = [w for w in text_raw.split(' ') if w.strip() != '' or w == '\n']
    #print ('length: ', len(sample_in_words))
    
    # calculating the word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # set the min word frequency to 1 for now
    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)
      
    words = set(text_in_words)
    words = sorted(set(words) - ignored_words)
    
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))
    
    # sequencing the text
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add sequences where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored+1
    
    # split into train and test data
    (sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)
    
#    # Building the model
#    model = Sequential()
#    model.add(LSTM(128, input_shape=(SEQUENCE_LEN, len(words))))
#    model.add(Dense(len(words)))
#    model.add(Activation('softmax'))
    
    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
        len(words),
        SEQUENCE_LEN,
        MIN_WORD_FREQUENCY
    )
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20)
    callbacks_list = [checkpoint, print_callback, early_stopping]
    
    filename = "generated_text_" + str(time.time())
    examples_file = open("./gentext/"+filename, "w")
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=100,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                        validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)