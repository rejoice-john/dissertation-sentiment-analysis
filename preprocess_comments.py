'''
Comment preprocessing pipeline (minimal).

Reads each series' raw comment CSV and strips URLs, since VADER, TextBlob,
and AFINN read punctuation, capitalisation, and words like 'not' as
sentiment signals and lose accuracy if those get stripped out first.
'''

import os
import re
import csv

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'processed')

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]


def strip_urls(text):
    # Removes URLs and collapses any extra whitespace left behind,
    # leaving punctuation, capitalisation, and words intact
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_file(series):
    # Reads one series' raw comment CSV and adds a URL-stripped text column
    input_path = os.path.join(INPUT_DIR, series + '.csv')
    if not os.path.exists(input_path):
        print('No raw file found for ' + series + ', skipping.')
        return []

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get('text', '')
            row['text_no_urls'] = strip_urls(original)
            rows.append(row)

    print('Processed ' + str(len(rows)) + ' comments for ' + series)
    return rows


def write_csv(path, rows):
    # Writes the processed rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- strip URLs from comments for all series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
master_rows = []

for series in SERIES_NAMES:
    rows = process_file(series)
    write_csv(os.path.join(OUTPUT_DIR, series + '_processed.csv'), rows)
    master_rows.extend(rows)

write_csv(os.path.join(OUTPUT_DIR, 'all_series_processed.csv'), master_rows)
print('Done. ' + str(len(master_rows)) + ' total comments processed and written to ' + OUTPUT_DIR)