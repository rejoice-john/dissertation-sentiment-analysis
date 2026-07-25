'''
Comment preprocessing pipeline.

Reads each series' raw comment CSV from the collection stage and produces
a cleaned, tokenised, and lemmatised version alongside the original text.
'''

import os
import re
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Downloads the NLTK resources needed below (skips any already installed)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

