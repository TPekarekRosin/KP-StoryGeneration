import collections
import re
import sys


def _preprocess(text):
    # First, make all the text lowercase.
    text = text.lower()

    # Bring all words separated across lines back together.
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

    # Separate single-quotes into their own words if
    # they appear at the beginning of an existing word.
    _words = []
    for word in words:
        for w in re.split(r'(^\')', word):
            if w != '':
                _words.append(w)
    words = _words

    # Separate single-quotes into their own words if
    # they appear at the end of an existing word.
    _words = []
    for word in words:
        for w in re.split(r'(\'$)', word):
            if w != '':
                _words.append(w)
    words = _words

    # Return the preprocessed list of words from the text.
    return words


def _postprocess(words):
    # Join all the words with spaces.
    text = " ".join(words)

    # Remove all spaces preceding end-punctuation.
    text = re.sub(r'( )([^\w])', r'\2', text)

    # Remove all spaces following a double or single quote.
    text = re.sub(r'([\"\'])( )', r'\1', text)

    # Capitalize the very first 'word' character found in the text.
    text = re.sub(r'(\w)', lambda x: x.group(1).upper(), text, count=1)

    # Capitalize the first 'word' character following periods,
    # question marks, and exclamation points.
    text = re.sub(
        r'([\.\!\?])([^\w]*)(\w)',
        lambda x: x.group(1) + x.group(2) + x.group(3).upper(),
        text)

    # Capitalize all standalone instances of 'i'.
    text = re.sub(r'(\s+i\s+)', lambda x: x.group(1).upper(), text)

    return text


def _build_dataset(words):
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


def build_vocabulary(input_filenames):
    words = []
    for filename in input_filenames:
        with open(filename, encoding="ISO-8859-1") as file:
            words.extend(_preprocess(file.read()))

    indices, count, dictionary, reverse_dictionary = _build_dataset(words)
    return words, indices, count, dictionary, reverse_dictionary


if __name__ == "__main__":
    input_filenames = sys.argv[1:]
    words, indices, count, dictionary, reverse_dictionary = build_vocabulary(input_filenames)

    print(dictionary)
