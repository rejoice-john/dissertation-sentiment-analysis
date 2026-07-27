'''
Comment preprocessing pipeline.

Reads each series' raw comment CSV, strips URLs, and filters out comments
detected as non-English before sentiment scoring, since VADER, TextBlob,
and AFINN are all English-lexicon tools and would silently score
non-English text as neutral rather than genuinely analysing it.

Language filtering uses a two-tier check validated by spot-checking
against this dataset:
  1. Any text containing non-Latin-script characters (Cyrillic, Arabic,
     Hebrew, Greek, Devanagari, Bengali, Thai, CJK, Korean Hangul) is
     flagged as non-English outright, since script alone is an
     unambiguous signal regardless of comment length.
  2. Latin-script text is only flagged as non-English if the comment is
     reasonably long (>= MIN_WORDS words) AND the detector's confidence
     in its top-choice language is high (>= MIN_CONFIDENCE). Short
     Latin-script reactions (e.g. "Can't wait!!") were found in spot
     checks to be misclassified into other Latin-script languages far
     too often to trust on their own, so anything below that bar is
     kept as English.

Excluded comments are written to a separate file per series rather than
discarded, so the exclusion count and reasoning can be reported and
audited in the methodology.
'''

import os
import re
import csv
from lingua import LanguageDetectorBuilder

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'processed')
EXCLUDED_DIR = os.path.join(INPUT_DIR, 'excluded_non_english')

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

MIN_WORDS = 8            # below this, Latin-script detection is too unreliable to trust
MIN_CONFIDENCE = 0.85    # lingua's confidence in its top-choice language
PROGRESS_EVERY = 500     # print a progress line every 500 comments while processing

detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

NON_LATIN_PATTERN = re.compile(
    r'[\u0400-\u052F'    # Cyrillic
    r'\u0600-\u06FF\u0750-\u077F'   # Arabic
    r'\u0590-\u05FF'     # Hebrew
    r'\u0370-\u03FF'     # Greek
    r'\u0900-\u097F'     # Devanagari
    r'\u0980-\u09FF'     # Bengali
    r'\u0E00-\u0E7F'     # Thai
    r'\u4E00-\u9FFF\u3040-\u30FF'   # CJK / Japanese kana
    r'\uAC00-\uD7AF'     # Korean Hangul
    r']'
)


def strip_urls(text):
    # Removes URLs and collapses any extra whitespace left behind,
    # leaving punctuation, capitalisation, and words intact
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_non_latin_script(text):
    return bool(NON_LATIN_PATTERN.search(text))


def classify_language(text):
    # Returns (is_non_english, detected_language, reason)
    if not text or not text.strip():
        return False, 'unknown', 'empty'

    if has_non_latin_script(text):
        language = detector.detect_language_of(text)
        code = language.iso_code_639_1.name if language else 'unknown'
        return True, code, 'non_latin_script'

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return False, 'EN', 'too_short_to_trust'

    confidence_values = detector.compute_language_confidence_values(text)
    if not confidence_values:
        return False, 'EN', 'no_confident_result'

    top = confidence_values[0]
    if top.value < MIN_CONFIDENCE:
        return False, 'EN', 'low_confidence'

    code = top.language.iso_code_639_1.name
    if code == 'EN':
        return False, 'EN', 'confident_english'

    return True, code, 'confident_non_english'


def process_file(series):
    # Reads one series' raw comment CSV, strips URLs, and separates
    # comments into kept (English) and excluded (non-English) lists
    input_path = os.path.join(INPUT_DIR, series + '.csv')
    if not os.path.exists(input_path):
        print('No raw file found for ' + series + ', skipping.')
        return [], []

    kept_rows = []
    excluded_rows = []
    urls_stripped = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            original = row.get('text', '')
            cleaned = strip_urls(original)
            if cleaned != original:
                urls_stripped += 1
            row['text_no_urls'] = cleaned

            is_non_english, lang, reason = classify_language(cleaned)
            if is_non_english:
                row['detected_language'] = lang
                row['flag_reason'] = reason
                excluded_rows.append(row)
            else:
                kept_rows.append(row)

            if i % PROGRESS_EVERY == 0:
                print('  ...checked ' + str(i) + ' comments so far ('
                      + str(len(excluded_rows)) + ' excluded so far)')

    print('  ' + series + ': ' + str(len(kept_rows) + len(excluded_rows)) + ' comments read, '
          + str(urls_stripped) + ' contained a URL that was stripped, '
          + str(len(excluded_rows)) + ' excluded as non-English')
    return kept_rows, excluded_rows


def write_csv(path, rows):
    # Writes rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- strip URLs and filter non-English comments for all series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXCLUDED_DIR, exist_ok=True)
master_rows = []
master_excluded = []

print('Preprocessing comments for all series...')
print('')

for series in SERIES_NAMES:
    print('--- ' + series + ' ---')
    kept, excluded = process_file(series)
    write_csv(os.path.join(OUTPUT_DIR, series + '_processed.csv'), kept)
    write_csv(os.path.join(EXCLUDED_DIR, series + '_excluded.csv'), excluded)
    master_rows.extend(kept)
    master_excluded.extend(excluded)
    print('  Written ' + series + '_processed.csv (' + str(len(kept)) + ' rows)')
    print('')

write_csv(os.path.join(OUTPUT_DIR, 'all_series_processed.csv'), master_rows)
write_csv(os.path.join(EXCLUDED_DIR, 'all_series_excluded.csv'), master_excluded)
print('Done. ' + str(len(master_rows)) + ' comments kept, ' + str(len(master_excluded))
      + ' excluded as non-English, written to ' + OUTPUT_DIR)