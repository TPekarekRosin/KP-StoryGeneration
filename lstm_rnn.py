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

from keras.callbacks import LambdaCallback
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.utils import np_utils

from utils import build_vocabulary


# SET CONSTANTS
# All of these paramters are tunable for experimentation.
# The values chosen below are set to reflect training on a GPU.
SEQUENCE_LEN = 20 # number of words used in the seeded sequence
STEP = 1 # increment by a number of words when sequencing the text
PERCENTAGE_TO_TEST = 10 # percentage of the input to test the model on
NUM_EPOCHS = 1000 # number of epochs to run our model for
BATCH_SIZE = 8192 # batch size of the data to run our model over

# Build path names to local folders for any generated files.
GENTEXT_FOLDER = os.path.join(os.path.dirname(__file__), "gentext")
PLOTS_FOLDER = os.path.join(os.path.dirname(__file__), "plots")
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "models")

# Timestamp used for any generated files.
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

WORD2VEC_PATH = MODELS_FOLDER + "/word2vec_model_2018-12-06-222657.h5"


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


def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    # TODO exchange input_dim and output_dim to CONSTANTS
    # The 'input_dim' is the size of the 'word_indices' dictionary.
    # The 'output_dim' is the output dimension of the word2vec model.
    # The name 'embedding' comes from the name for the layer used in the
    # original word2vec model.
    model.add(Embedding(input_dim=17961, output_dim=300, name='embedding'))
    model.add(Bidirectional(LSTM(128), name='lstm'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(word_indices),name='dense'))
    model.add(Activation('softmax'))
    # adds the pretrained weights to the embedding layer by name
    model.load_weights(WORD2VEC_PATH, by_name=True)
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


def plot_accuracy(results):
    plt.clf()

    # plot the accuracy of the model and save it to file
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    path_acc = os.path.join(PLOTS_FOLDER, "lstm_plot_acc_" + TIMESTAMP)

    plt.savefig(path_acc, bbox_inches='tight')


def plot_loss(results):
    plt.clf()

    # plot the loss of the model and save it to file
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    path_loss = os.path.join(PLOTS_FOLDER, "lstm_plot_loss_" + TIMESTAMP)

    plt.savefig(path_loss, bbox_inches='tight')


if __name__ == "__main__":
    # Create folders for any generated files.
    os.makedirs(GENTEXT_FOLDER, exist_ok=True)
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    # PREPROCESS THE DATA
    # pass in the text file name as the first argument
    # e.g. `$ python lstm_rnn.py sample1.txt`
    input_filenames = sys.argv[1:]
    words, indices, count, word_indices, indices_word = build_vocabulary(input_filenames)

    # SEQUENCE THE TEXT
    sequences = []
    next_words = []
    for i in range(0, len(words) - SEQUENCE_LEN, STEP):
        sequences.append(words[i:i+SEQUENCE_LEN])
        next_words.append(words[i+SEQUENCE_LEN])

    # SPLIT DATA INTO TRAIN AND TEST DATA
    (sequences_train, next_words_train), (sequences_test, next_words_test) = shuffle_and_split_training_set(sequences, next_words)

    # BUILD AND COMPILE THE MODEL
    model = get_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    # SET A CALLBACK TO PRINT SAMPLE OUTPUT DURING TRAINING
    gentext_callback = LambdaCallback(on_epoch_end=gentext)

    # SET THE TRAINING PARAMETERS, THEN FIT THE MODEL
    gentext_filename = os.path.join(GENTEXT_FOLDER, "lstm_" + TIMESTAMP)
    gentext_file = open(gentext_filename, "w")
    results = model.fit_generator(
        epochs=NUM_EPOCHS,
        generator=generator(sequences_train, next_words_train),
        steps_per_epoch=int(len(sequences_train)/BATCH_SIZE) + 1,
        validation_data=generator(sequences_test, next_words_test),
        validation_steps=int(len(sequences_test)/BATCH_SIZE) + 1,
        callbacks=[gentext_callback])
    gentext_file.close()

    # serialize model to JSON
    saved_model_prefix = os.path.join(MODELS_FOLDER, "lstm_model_" + TIMESTAMP)
    model_json = model.to_json()
    with open(saved_model_prefix + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(saved_model_prefix + ".h5")
    print("Saved model to disk")

    # visualization
    plot_accuracy(results)
    plot_loss(results)
