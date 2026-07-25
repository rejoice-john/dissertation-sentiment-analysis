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

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'processed')

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(tag):
    # Maps NLTK's POS tag letters onto the tags WordNetLemmatizer expects,
    # so a word like 'running' lemmatises correctly as a verb instead of
    # defaulting to being treated as a noun
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('R'):
        return 'r'
    return 'n'


def clean_text(text):
    # Lowercases the text, strips URLs, and removes anything that isn't
    # a letter or whitespace (numbers, punctuation, emojis)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenise_and_lemmatise(text):
    # Splits cleaned text into words, drops stopwords and single-letter
    # tokens, and reduces each remaining word to its dictionary form
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = []
    for word, tag in tagged:
        if word not in STOP_WORDS and len(word) > 1:
            wn_tag = get_wordnet_pos(tag)
            lemmas.append(LEMMATIZER.lemmatize(word, wn_tag))
    return lemmas
