#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:49:38 2018

@author: Amy Bryce, Theresa Pekarek-Rosin

Inspired by enrique a.:
    Blog: https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
    Source code: https://github.com/enriqueav/lstm_lyrics

Inspiration for next steps:
https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3
"""

import re
import sys
import time

import numpy as np

from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.utils import np_utils


# SET CONSTANTS
MIN_WORD_FREQUENCY = 1 # how often a word appears in the original text
SEQUENCE_LEN = 10 # number of words used in the seeded sentence
STEP = 1 # increment by a number of words when sequencing the text
BATCH_SIZE = 32 #


def preprocess(text):
    # First, bring all words split across lines back together.
    # This also accounts for multiple new lines and indents between when the
    # word on the first line ends and the rest of the word is later on.
    # e.g. "exagg-\n\n    erated" --> "exaggerated\n\n    "
    text = re.sub(r'(.*)-(\n\s*)([^\s]+)(.*)', r'\1\3\2\4', text)

    # Now split the text into words using spaces as a delimter.
    words = text.split(' ')

    # Now search through each 'word' and turn any empty strings into
    # a space (we keep these to indicate indenting paragraphs, quotations,
    # etc. for style detection).
    _words = []
    for word in words:
        if word == '':
            _words.append(' ')
        else:
            _words.append(word)
    words = _words

    # Now search through each 'word' and find instances where a
    # paragraph had existed (i.e. where there is now a double '\n\n').
    # e.g. 'word\n\nword'. Replace these instances with a separate
    # word of just '\n'.
    _words = []
    for word in words:
        while '\n\n' in word:
            w, word = word.split('\n\n', 1)
            if w != '':
                _words.append(w)
            _words.append('\n')
        if word != '':
            _words.append(word)
    words = _words

    # Now split any remaining words that have '\n' in them into two
    # separate words.
    _words = []
    for word in words:
        if '\n' in word and '\n' != word:
            for w in word.split('\n'):
                if w != '':
                    _words.append(w)
        else:
            _words.append(word)
    words = _words

    # Now split each word to pull out any punctutation into its own word.
    _words = []
    for word in words:
        for w in re.split("([^\w\'\-])", word):
            if w != '':
                _words.append(w)
    words = _words

    # Return the preprocessed list of words from the text.
    return words


# TODO - EXPERIMENT: try using different percentages of train and test data
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=10):
    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    return (x_train, y_train), (x_test, y_test)


# Use a data generator to feed the model with chunks of the training set,
# one for each batch, instead of feeding everything at once.
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


# TODO - EXPERIMENT: try different network structures and parameters
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
    # PREPROCESS THE DATA
    # pass in the text file name as the first argument
    # e.g. `$ python lstm_rnn.py sample1.txt`
    filename = sys.argv[1]
    with open(filename) as file:
        text_in_words = preprocess(file.read())

    # calculate the word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # set the minimum word frequency
    # TODO - EXPERIMENT: try using different frequencies
    # create a set of ignored words that don't meet the minimum word frequency
    ignored_words = set() # set(): an unordered list of unique elements
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    # create a set of words parsed from the text
    words = set(text_in_words)
    # remove ignored words from the word set, then sort the set
    words = sorted(set(words) - ignored_words)
    
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # SEQUENCE THE TEXT
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP): # TODO: STEP could be hard-coded to = 1
        # Only add sequences to the sentences list where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1

    # SPLIT DATA INTO TRAIN AND TEST DATA
    (sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(sentences, next_words)

    # BUILD AND COMPILE THE MODEL
    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    # CREATE CALLBACKS FOR WHEN WE RUN THE MODEL
    # set the file path for storing the output from the model
    file_path = "./checkpoints/LSTM_Sherlock-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % (
        len(words),
        SEQUENCE_LEN,
        MIN_WORD_FREQUENCY
    )
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True) # save the weights every epoch
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20) # halt the training if there no gain in the loss in 5 epochs
    callbacks_list = [checkpoint, print_callback, early_stopping]

    # SET THE TRAINING PARAMETERS, THEN FIT THE MODEL
    # TODO - EXPERIMENT: try training with different # of batch sizes and epochs
    filename = "generated_text_" + str(time.time())
    examples_file = open("./gentext/"+filename, "w")
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=100,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                        validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)