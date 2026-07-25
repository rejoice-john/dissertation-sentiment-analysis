'''
Sentiment scoring pipeline.

Runs VADER, TextBlob, and AFINN on each comment's original (URL-stripped)
text and records each tool's score plus a positive/negative/neutral label,
so the three tools can be directly compared against the same dataset.
'''

import os
import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn

nltk.download('vader_lexicon', quiet=True)

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\processed'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

# VADER's compound score and TextBlob's polarity both sit on a -1 to 1
# scale, so the same small buffer around zero keeps their classification
# consistent for comparison. AFINN scores are unbounded integer sums, not
# a -1 to 1 scale, so it uses a plain zero cutoff instead.
BOUNDED_THRESHOLD = 0.05

vader = SentimentIntensityAnalyzer()
afinn = Afinn()


def classify_bounded(score):
    # Used for VADER compound and TextBlob polarity (-1 to 1 scale)
    if score >= BOUNDED_THRESHOLD:
        return 'positive'
    if score <= -BOUNDED_THRESHOLD:
        return 'negative'
    return 'neutral'


def classify_afinn(score):
    # AFINN scores are unbounded integer sums, so zero is the natural cutoff
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    return 'neutral'


def score_comment(text):
    # Runs all three tools on one comment and returns their scores and labels
    vader_scores = vader.polarity_scores(text)
    vader_compound = vader_scores['compound']

    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity

    afinn_score = afinn.score(text)

    return {
        'vader_compound': vader_compound,
        'vader_label': classify_bounded(vader_compound),
        'textblob_polarity': textblob_polarity,
        'textblob_subjectivity': textblob_subjectivity,
        'textblob_label': classify_bounded(textblob_polarity),
        'afinn_score': afinn_score,
        'afinn_label': classify_afinn(afinn_score),
    }


def process_file(series):
    # Reads one series' preprocessed CSV and scores each comment with all three tools
    input_path = os.path.join(INPUT_DIR, series + '_processed.csv')
    if not os.path.exists(input_path):
        print('No processed file found for ' + series + ', skipping.')
        return []

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text_no_urls', row.get('text', ''))
            scores = score_comment(text)
            row.update(scores)
            rows.append(row)

    print('Scored ' + str(len(rows)) + ' comments for ' + series)
    return rows


def write_csv(path, rows):
    # Writes the scored rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- score comments for all series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
master_rows = []

for series in SERIES_NAMES:
    rows = process_file(series)
    write_csv(os.path.join(OUTPUT_DIR, series + '_scored.csv'), rows)
    master_rows.extend(rows)

write_csv(os.path.join(OUTPUT_DIR, 'all_series_scored.csv'), master_rows)
print('Done. ' + str(len(master_rows)) + ' total comments scored and written to ' + OUTPUT_DIR)