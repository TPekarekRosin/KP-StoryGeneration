# KP-StoryGeneration: Sherlock Holmes
Created by [Amy Bryce](https://github.com/AmyBryce) & [Theresa Pekarek-Rosin](https://github.com/TPekarekRosin).\
Github Repository: <https://github.com/TPekarekRosin/KP-StoryGeneration>

### Setup Instructions

##### Code Prerequisites
Install python 3.6.

##### Set up python environment:
(1) Run `$ ./bootstrap.sh`.\
(2) Start the virtual environment with `$ source activate`.\
You should now be ready to run your code.\
(3) Run `$ deactivate` when you are done.

##### Run the python code:
After you've started the virtual environment with `$ source activate`, you can run your python code (e.g. `$ python helloworld.py`).

##### Troubleshooting:
- If you are given a permissions error when trying to run `$ ./bootstrap.sh` (e.g. `-bash: ./bootstrap.sh: Permission denied`), then run this command to add executable permissions: `$ chmod a+x bootstrap.sh`.

### Inference: Using the Story Generator
(1) In the terminal, run `$ python lstm_infer.py`.\
(2) You will then be prompted with:
```
How would you like to start your Sherlock Holmes story?
Please input up to 20 'words' (note that all punctuation will also be considered a 'word').
Type your words here and press <ENTER> when you are done:
```
(3) Please enter a seed sentence that is up to 20 words. For example: `It was the best of times, it was the worst of times`. If you enter a seed sentence that is longer than 20 words, the Story Generator will ignore the excess words.\
(4) Once you hit `<ENTER>`, the Story Generator will output a unique Sherlock Holmes story, based on your starting sentence, onto your terminal screen.
Please note that this current version of the Story Generator will produce the same story given the same seed sentence every time.\
(5) The saved models used for the current version of the Story Generator can be found in the [models/](https://github.com/TPekarekRosin/KP-StoryGeneration/tree/master/models) directory.
