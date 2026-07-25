'''
Dataset overview pipeline.

Produces a single summary table across all series: how many comments were
actually collected, what date range each series' window covered, and the
overall sentiment split per tool. Useful as an at-a-glance check and as a
starting table for the Results chapter.
'''

import os
import csv
from datetime import datetime

SCORED_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\overview'

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]


def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def load_series(series):
    # Reads one series' scored CSV
    input_path = os.path.join(SCORED_DIR, series + '_scored.csv')
    if not os.path.exists(input_path):
        print('No scored file found for ' + series + ', skipping.')
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def summarise(series, rows):
    # Builds one summary row covering counts, date range, and sentiment
    # split for a series (or the whole dataset if series is 'ALL_SERIES')
    if not rows:
        return None

    top_level = [r for r in rows if not r.get('parent_id')]
    replies = [r for r in rows if r.get('parent_id')]
    dates = [parse_time(r['published_at']) for r in rows]

    summary = {
        'series': series,
        'total_top_level_comments': len(top_level),
        'total_replies': len(replies),
        'total_all_rows': len(rows),
        'earliest_comment_date': min(dates).date(),
        'latest_comment_date': max(dates).date(),
    }

    for tool, label_field in [
        ('vader', 'vader_label'),
        ('textblob', 'textblob_label'),
        ('afinn', 'afinn_label'),
    ]:
        total = len(rows)
        positive = sum(1 for r in rows if r.get(label_field) == 'positive')
        negative = sum(1 for r in rows if r.get(label_field) == 'negative')
        neutral = sum(1 for r in rows if r.get(label_field) == 'neutral')

        summary[tool + '_positive_pct'] = round(positive / total * 100, 2)
        summary[tool + '_negative_pct'] = round(negative / total * 100, 2)
        summary[tool + '_neutral_pct'] = round(neutral / total * 100, 2)

    return summary


def write_csv(path, rows):
    # Writes the summary rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- build a summary row per series, plus one for the whole dataset ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
summaries = []
all_rows = []

for series in SERIES_NAMES:
    rows = load_series(series)
    if not rows:
        continue
    all_rows.extend(rows)
    summary = summarise(series, rows)
    if summary:
        summaries.append(summary)
        print('Summarised ' + series)

overall_summary = summarise('ALL_SERIES_COMBINED', all_rows)
if overall_summary:
    summaries.append(overall_summary)

write_csv(os.path.join(OUTPUT_DIR, 'dataset_overview.csv'), summaries)
print('Done. Overview written to ' + OUTPUT_DIR)