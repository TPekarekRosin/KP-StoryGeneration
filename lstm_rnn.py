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

from utils import preprocess


# SET CONSTANTS
# All of these paramters are tunable for experimentation.
SEQUENCE_LEN = 10 # number of words used in the seeded sequence
STEP = 1 # increment by a number of words when sequencing the text
PERCENTAGE_TO_TEST = 10 # percentage of the input to test the model on
NUM_EPOCHS = 100 # number of epochs to run our model for
BATCH_SIZE = 32 # batch size of the data to run our model over

# Build path names to local folders for any generated files.
CHECKPOINTS_FOLDER = os.path.join(os.path.dirname(__file__), "checkpoints")
GENTEXT_FOLDER = os.path.join(os.path.dirname(__file__), "gentext")
PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "plots")

# Timestamp used for any generated files.
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


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


# Use a data generator to feed the model with chunks of the training and test
# sets, one for each batch, instead of feeding everything at once.
def generator(sequences, next_words):
    index = 0
    while True:
        x = np.zeros((BATCH_SIZE, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((BATCH_SIZE), dtype=np.int32)
        for i in range(BATCH_SIZE):
            for j, w in enumerate(sequences[index % len(sequences)]):
                x[i, j] = word_indices[w]
            y[i] = word_indices[next_words[index % len(sequences)]]
            index = index + 1
        yield x, y


# TODO - EXPERIMENT: try different network structures and parameters
def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(vocabulary), output_dim=1024)) #turns indexes into dense vectors of fixed size
    model.add(Bidirectional(LSTM(128)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(vocabulary)))
    model.add(Activation('softmax'))
    return model


def gentext(epoch, logs):
    def get_next_index(preds, temperature=1.0):
        # helper function to sample a random index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # Function invoked at end of each epoch. Prints generated text.
    gentext_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(sequences))
    seed = sequences[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sequence = seed
        gentext_file.write('----- Diversity:' + str(diversity) + '\n')
        gentext_file.write('----- Generating with seed:\n"' + ' '.join(sequence) + '"\n')
        gentext_file.write(' '.join(sequence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN))
            for t, word in enumerate(sequence):
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = get_next_index(preds, diversity)
            next_word = indices_word[next_index]

            sequence = sequence[1:]
            sequence.append(next_word)

            gentext_file.write(" "+next_word)
        gentext_file.write('\n')
    gentext_file.write('='*80 + '\n')
    gentext_file.flush()


def plot_accuracy(results, input_filename):
    plt.clf()

    # plot the accuracy of the model and save it to file
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    path_acc = os.path.join(PLOTS_FOLDER, "acc_" + re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)

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

    path_loss = os.path.join(PLOTS_FOLDER, "loss_" + re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)

    plt.savefig(path_loss, bbox_inches='tight')


if __name__ == "__main__":
    # Create folders for any generated files.
    os.makedirs(CHECKPOINTS_FOLDER, exist_ok=True)
    os.makedirs(GENTEXT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    # PREPROCESS THE DATA
    # pass in the text file name as the first argument
    # e.g. `$ python lstm_rnn.py sample1.txt`
    input_filename = sys.argv[1]
    with open(input_filename) as file:
        words_in_text = preprocess(file.read())

    # SEQUENCE THE TEXT
    sequences = []
    next_words = []
    for i in range(0, len(words_in_text) - SEQUENCE_LEN, STEP):
        sequences.append(words_in_text[i:i+SEQUENCE_LEN])
        next_words.append(words_in_text[i+SEQUENCE_LEN])

    # Condense the fully tokenized text into a 'set' of unique words
    # and build a set of indices into it.
    vocabulary = set(words_in_text)
    word_indices = dict((w, i) for i, w in enumerate(vocabulary))
    indices_word = dict((i, w) for i, w in enumerate(vocabulary))

    # SPLIT DATA INTO TRAIN AND TEST DATA
    (sequences_train, next_words_train), (sequences_test, next_words_test) = shuffle_and_split_training_set(sequences, next_words)

    # BUILD AND COMPILE THE MODEL
    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    # CREATE CALLBACKS FOR WHEN WE RUN THE MODEL
    # set the file path for storing the output from the model
    checkpoint_filename = os.path.join(CHECKPOINTS_FOLDER, "LSTM_Sherlock-epoch{epoch:03d}-vocabulary%d-sequence%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}") % (
        len(vocabulary),
        SEQUENCE_LEN
    )
    checkpoint_callback = ModelCheckpoint(checkpoint_filename, monitor='val_acc', save_best_only=True) # save the weights every epoch
    gentext_callback = LambdaCallback(on_epoch_end=gentext)
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=20) # halt the training if there no gain in the loss in 5 epochs

    # SET THE TRAINING PARAMETERS, THEN FIT THE MODEL
    # TODO - EXPERIMENT: try training with different # of batch sizes and epochs
    gentext_filename = os.path.join(GENTEXT_FOLDER, re.sub('\.txt$', '', input_filename) + "_" + TIMESTAMP)
    gentext_file = open(gentext_filename, "w")
    results = model.fit_generator(
        epochs=NUM_EPOCHS,
        generator=generator(sequences_train, next_words_train),
        steps_per_epoch=int(len(sequences_train)/BATCH_SIZE) + 1,
        validation_data=generator(sequences_test, next_words_test),
        validation_steps=int(len(sequences_test)/BATCH_SIZE) + 1,
        callbacks=[checkpoint_callback, gentext_callback, early_stopping_callback])
    gentext_file.close()

    # visualization
    plot_accuracy(results, input_filename)
    plot_loss(results, input_filename)
