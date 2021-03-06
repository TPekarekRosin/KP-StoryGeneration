import glob
import h5py
import json
import os
import random
import sys

import numpy as np

from keras.models import model_from_json

from utils import _postprocess, _preprocess, build_vocabulary


DATA_FOLDER = os.path.join(os.path.dirname(__file__), "Data")
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "models")
LSTM_MODEL_PATH = MODELS_FOLDER + "/lstm_model_2018-12-12-005552.json"
LSTM_WEIGHTS_PATH = MODELS_FOLDER + "/lstm_model_2018-12-12-005552.h5"

SEQUENCE_LEN = 20
NUM_PERIODS_UNTIL_STOP = 50


if __name__ == "__main__":
    # Load the LSTM model.
    with open(LSTM_MODEL_PATH, "r") as json_file:
        lstm = model_from_json(json_file.read())
        lstm.load_weights(LSTM_WEIGHTS_PATH)

    # Build out our input vocabulary from the set of 'Data' files.
    input_filenames = glob.glob(DATA_FOLDER + "/*")
    _, _, _, dictionary, reverse_dictionary = build_vocabulary(input_filenames)

    # Capture an initial input sequence of 20 words or less.
    print("\n\n\n\n")
    print("How would you like to start your Sherlock Holmes story?")
    print("Please input up to 20 'words' (note that all punctuation will also be considered a 'word').")
    print("Type your words here and press <ENTER> when you are done:")
    print(">>> ", end="", flush=True)
    input_seq = _preprocess(sys.stdin.readline())
    print("\nProcessing...")

    # Copy the input sequence as the initial output (capping at SEQUENCE_LEN).
    output = input_seq[:SEQUENCE_LEN]

    # Convert the input sequence (represented as the copied output) into a
    # list of indices appropriate for passing to the LSTM model as input.
    lstm_input = np.zeros((1, SEQUENCE_LEN))
    for i, word in enumerate(output):
        # Adjust the index to account for input
        # sequences shorter than SEQUENCE_LEN.
        adjusted_index = SEQUENCE_LEN - len(output) + i
        if word not in dictionary:
            lstm_input[0, adjusted_index] = 0
        else:
            lstm_input[0, adjusted_index] = dictionary[word]

    # Start predicting new words from the input
    # sequence and adding them to the output.
    num_periods = 0
    while num_periods < NUM_PERIODS_UNTIL_STOP:
       lstm_output = lstm.predict(lstm_input)[0]

       # Select the top predicted word.
       next_word_index = lstm_output.argmax()

       # Get the next input sequence by shifting out the first word
       # and appending the most recently chosen word.
       new_lstm_input = np.append(lstm_input[0][1:], next_word_index)

       # Break early if the new input sequence is identical to the previous one.
       # If we didn't do this, we would be stuck in the same sequence forever.
       if (new_lstm_input == lstm_input[0]).all():
           break

       # Otherwise update the input sequence and continue.
       lstm_input[0] = new_lstm_input

       # Look up the next word in the dictionary and append it to the output.
       next_word = reverse_dictionary[next_word_index]
       output.append(next_word)

       if next_word == ".":
           num_periods += 1

    # Print the output.
    print(_postprocess(output))
