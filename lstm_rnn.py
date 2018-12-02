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

import os
import re
import sys
import datetime

import numpy as np

import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential
from keras.utils import np_utils


# SET CONSTANTS
SEQUENCE_LEN = 10 # number of words used in the seeded sequence
STEP = 1 # increment by a number of words when sequencing the text
PERCENTAGE_TO_TEST = 10 # percentage of the input to test the model on
BATCH_SIZE = 32 #

# Build path names to local folders for any generated files.
CHECKPOINTS_FOLDER = os.path.join(os.path.dirname(__file__), "checkpoints")
GENTEXT_FOLDER = os.path.join(os.path.dirname(__file__), "gentext")
PLOT_FOLDER = os.path.join(os.path.dirname(__file__), "plots")

# Timestamp used for any generated files.
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def preprocess(text):
    # First, bring all words separated across lines back together.
    # This also accounts for multiple new lines and indents between when the
    # word on the first line ends and the rest of the word is later on
    # (e.g. "exagg-\n\n    erated" --> "exaggerated\n\n    ").
    text = re.sub(r'(.*)-(\n\s*)([^\s]+)', r'\1\3\2', text)

    # Use regular expressions to split the text by \n followed by at least one
    # whitespace character; this allows us to keep the \n's and subsequent
    # whitespace characters. Essentially, we are splitting the text into
    # paragraph strings.
    _words = []
    for w in re.split(r'(\n\s+)', text):
        if w != '':
            _words.append(w)
    words = _words

    # Use regular expressions to split the list of words (i.e. paragraphs) by
    # any whitespaces (e.g. \n, space, etc.) that exist between non-whitespace
    # characters and remove them. Essentially, we are splitting the paragraphs
    # into word strings.
    _words = []
    for word in words:
        for w in re.split(r'([^\s]+)\s+', word):
            if w != '':
                _words.append(w)
    words = _words

    # Use regular expressions to split the list of words by any remaining
    # whitespace characters. This will help to preserve any notion of
    # indentations or other special formatting found in the orginal text
    # (i.e. some whitespace will be treated as word strings).
    _words = []
    for word in words:
        for w in re.split(r'(\s)', word):
            if w != '':
                _words.append(w)
    words = _words

    # Use regular expressions to split the list of words by anything that is not
    # a letter, or number, or ', or - (i.e. we want to preserve hyphenated
    # words or words using an apostrophe). Therefore, we are splitting by any
    # form of punctuation.
    _words = []
    for word in words:
        for w in re.split(r'([^\w\'\-])', word):
            if w != '':
                _words.append(w)
    words = _words

    # Return the preprocessed list of words from the text.
    return words


# TODO - EXPERIMENT: try using different percentages of train and test data
def shuffle_and_split_training_set(sequences, next_words):
    _sequences = []
    _next_words = []
    for i in np.random.permutation(len(sequences)):
        _sequences.append(sequences[i])
        _next_words.append(next_words[i])

    cut_index = int(len(sequences) * (1.-(PERCENTAGE_TO_TEST/100.)))
    x_train, x_test = _sequences[:cut_index], _sequences[cut_index:]
    y_train, y_test = _next_words[:cut_index], _next_words[cut_index:]

    return (x_train, y_train), (x_test, y_test)


# Use a data generator to feed the model with chunks of the training set,
# one for each batch, instead of feeding everything at once.
def generator(sequences, next_words, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sequences[index % len(sequences)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_words[index % len(sequences)]]
            index = index + 1
        yield x, y


# TODO - EXPERIMENT: try different network structures and parameters
def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words_in_text), output_dim=1024)) #turns indexes into dense vectors of fixed size
    model.add(Bidirectional(LSTM(128)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words_in_text)))
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
    seed_index = np.random.randint(len(sequences+sequences_test))
    seed = (sequences+sequences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sequence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sequence) + '"\n')
        examples_file.write(' '.join(sequence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sequence):
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sequence = sequence[1:]
            sequence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()


def plot_accuracy(results, input_filename):
    plt.clf()

    # plot the accuracy of the model and save it to file
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    path_acc = os.path.join(PLOT_FOLDER, "acc_" + re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)

    plt.savefig(path_acc, bbox_inches='tight')


def plot_loss(results, input_filename):
    plt.clf()

    # plot the loss of the model and save it to file
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    path_loss = os.path.join(PLOT_FOLDER, "loss_" + re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)

    plt.savefig(path_loss, bbox_inches='tight')


if __name__ == "__main__":
    # Create folders for any generated files.
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
    os.makedirs(GENTEXT_FOLDER, exist_ok=True)
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    # PREPROCESS THE DATA
    # pass in the text file name as the first argument
    # e.g. `$ python lstm_rnn.py sample1.txt`
    input_filename = sys.argv[1]
    with open(input_filename) as file:
        words_in_text = preprocess(file.read())

    words_in_text = set(words_in_text)
    word_indices = dict((w, i) for i, w in enumerate(words_in_text))
    indices_word = dict((i, w) for i, w in enumerate(words_in_text))

    # SEQUENCE THE TEXT
    sequences = []
    next_words = []
    for i in range(0, len(words_in_text) - SEQUENCE_LEN, STEP):
        sequences.append(words_in_text[i:i+SEQUENCE_LEN])
        next_words.append(words_in_text[i+SEQUENCE_LEN])

    # SPLIT DATA INTO TRAIN AND TEST DATA
    (sequences, next_words), (sequences_test, next_words_test) = shuffle_and_split_training_set(sequences, next_words)

    # BUILD AND COMPILE THE MODEL
    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    # CREATE CALLBACKS FOR WHEN WE RUN THE MODEL
    # set the file path for storing the output from the model
    file_path = os.path.join(CHECKPOINTS_FOLDER, "LSTM_Sherlock-epoch{epoch:03d}-words%d-sequence%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}") % (
        len(words_in_text),
        SEQUENCE_LEN
    )
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True) # save the weights every epoch
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20) # halt the training if there no gain in the loss in 5 epochs
    callbacks_list = [checkpoint, print_callback, early_stopping]

    # SET THE TRAINING PARAMETERS, THEN FIT THE MODEL
    # TODO - EXPERIMENT: try training with different # of batch sizes and epochs
    gen_filename = os.path.join(GENTEXT_FOLDER, "gen_text_" + re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)
    examples_file = open(gen_filename, "w")
    results = model.fit_generator(generator(sequences, next_words, BATCH_SIZE),
                            steps_per_epoch=int(len(sequences)/BATCH_SIZE) + 1,
                            epochs=100,
                            callbacks=callbacks_list,
                            validation_data=generator(sequences_test, next_words_test, BATCH_SIZE),
                            validation_steps=int(len(sequences_test)/BATCH_SIZE) + 1)
    examples_file.close()

    # visualization
    plot_accuracy(results, input_filename)
    plot_loss(results, input_filename)
