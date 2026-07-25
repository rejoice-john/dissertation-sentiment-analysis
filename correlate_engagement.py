'''
Engagement correlation pipeline.

Tests the relationship between sentiment and engagement (likes) at two
levels: per-comment (large sample, strong statistical power) and
per-month (matches the "across the twelve-month window" framing in
objective 5, but with only 12 data points per series, results here are
indicative rather than statistically robust on their own).
'''

import os
import csv
from scipy.stats import spearmanr

SCORED_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'
MONTHLY_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\aggregated'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\correlation'

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

SENTIMENT_FIELDS = ['vader_compound', 'textblob_polarity', 'afinn_score']
ENGAGEMENT_FIELDS_MONTHLY = ['comment_volume', 'reply_activity', 'avg_likes']


def load_csv(path):
    # Reads a CSV file into a list of dictionaries
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def correlate_per_comment(rows, series):
    # Correlates each tool's sentiment score against each comment's own
    # like count, using every comment as one data point. Only "likes"
    # applies here -- volume and reply activity are group-level counts,
    # not something a single comment has on its own.
    results = []
    likes = [int(r.get('like_count', 0)) for r in rows]
    for field in SENTIMENT_FIELDS:
        scores = [float(r.get(field, 0)) for r in rows]
        rho, p_value = spearmanr(scores, likes)
        results.append({
            'series': series,
            'level': 'per_comment',
            'sentiment_field': field,
            'engagement_field': 'like_count',
            'n': len(rows),
            'spearman_rho': round(rho, 4),
            'p_value': round(p_value, 4),
        })
    return results


def correlate_monthly(rows, series):
    # Correlates each tool's average monthly sentiment score against
    # each monthly engagement figure, using one data point per month
    results = []
    for sentiment_field in SENTIMENT_FIELDS:
        tool = sentiment_field.split('_')[0]
        avg_field = tool + '_avg_score'
        scores = [float(r.get(avg_field, 0)) for r in rows]
        for engagement_field in ENGAGEMENT_FIELDS_MONTHLY:
            values = [float(r.get(engagement_field, 0)) for r in rows]
            rho, p_value = spearmanr(scores, values)
            results.append({
                'series': series,
                'level': 'monthly',
                'sentiment_field': avg_field,
                'engagement_field': engagement_field,
                'n': len(rows),
                'spearman_rho': round(rho, 4),
                'p_value': round(p_value, 4),
            })
    return results


def write_csv(path, rows):
    # Writes the correlation results out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- correlate sentiment and engagement at both levels, for all series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_results = []

for series in SERIES_NAMES:
    scored_rows = load_csv(os.path.join(SCORED_DIR, series + '_scored.csv'))
    monthly_rows = load_csv(os.path.join(MONTHLY_DIR, series + '_monthly.csv'))

    if scored_rows:
        all_results.extend(correlate_per_comment(scored_rows, series))
    if monthly_rows:
        all_results.extend(correlate_monthly(monthly_rows, series))

    print('Correlated ' + series)

write_csv(os.path.join(OUTPUT_DIR, 'engagement_correlation.csv'), all_results)
print('Done. Correlation results written to ' + OUTPUT_DIR)