from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import datetime
import os
import sys
import zipfile

import numpy as np
import tensorflow as tf

from utils import preprocess


MODELS_FOLDER = os.path.join(os.path.dirname(__file__), "models")

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")


def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    indices = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        indices.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return indices, count, dictionary, reversed_dictionary


def collect_data(input_filenames):
    words = []
    for filename in input_filenames:
        with open(filename, encoding="ISO-8859-1") as file:
            words.extend(preprocess(file.read()))

    print(words[:7])
    indices, count, dictionary, reverse_dictionary = build_dataset(words)
    del words  # Hint to reduce memory.
    return indices, count, dictionary, reverse_dictionary


input_filenames = sys.argv[1:]
indices, count, dictionary, reverse_dictionary = collect_data(input_filenames)
vocab_size = len(dictionary)
print(indices[:7])

window_size = 3
vector_dim = 300
epochs = 200000

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(indices, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity = merge.dot([target, context], axes=0, normalize=True)

# now perform the dot product operation to get a similarity measure
dot_product = merge.dot([target, context], axes=1, normalize=False)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)


class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()

# Create folders for any generated files.
os.makedirs(MODELS_FOLDER, exist_ok=True)
saved_model_prefix = os.path.join(MODELS_FOLDER, "word2vec_model_" + TIMESTAMP)

# serialize model to JSON
model_json = model.to_json()
with open(saved_model_prefix + ".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(saved_model_prefix + ".h5")
print("Saved model to disk")
