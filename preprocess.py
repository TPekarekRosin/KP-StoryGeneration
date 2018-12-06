import re
import sys


def preprocess(text):
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

    # Return the preprocessed list of words from the text.
    return words


if __name__ == "__main__":
    input_filename = sys.argv[1]
    with open(input_filename) as file:
        words_in_text = preprocess(file.read())

    print(words_in_text)
