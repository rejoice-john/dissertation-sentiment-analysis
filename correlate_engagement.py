'''
Engagement correlation pipeline.

Tests the relationship between sentiment and engagement (likes) at two
levels: per-comment (large sample, strong statistical power) and
per-month (matches the "across the twelve-month window" framing in
objective 5, but with only 12 data points per series, results here are
indicative rather than statistically robust on their own).

Since this runs many separate significance tests (60 in total across all
series/tools/levels), a Benjamini-Hochberg false discovery rate correction
is applied across all of them at the end -- without it, some "significant"
results would just be expected by chance from running so many tests.
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
ALPHA = 0.05   # significance threshold used both before and after correction


def load_csv(path):
    # Reads a CSV file into a list of dictionaries
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def benjamini_hochberg(p_values):
    # Adjusts a list of p-values for multiple testing using the
    # Benjamini-Hochberg false discovery rate procedure. Returns adjusted
    # p-values in the same order as the input list.
    m = len(p_values)
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda pair: pair[1])   # sort ascending by p-value

    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        original_index, p = indexed[rank - 1]
        candidate = p * m / rank
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min

    return adjusted


def correlate_per_comment(rows, series):
    # Correlates each tool's sentiment score against each comment's own
    # like count, using every comment as one data point
    results = []
    likes = [int(r.get('like_count', 0)) for r in rows]
    for field in SENTIMENT_FIELDS:
        scores = [float(r.get(field, 0)) for r in rows]
        rho, p_value = spearmanr(scores, likes)
        print('  [per_comment] ' + series + ' / ' + field + ' vs like_count: '
              + 'rho=' + str(round(rho, 4)) + ', p=' + str(round(p_value, 4))
              + ', n=' + str(len(rows)))
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
            print('  [monthly] ' + series + ' / ' + avg_field + ' vs ' + engagement_field + ': '
                  + 'rho=' + str(round(rho, 4)) + ', p=' + str(round(p_value, 4))
                  + ', n=' + str(len(rows)))
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

print('Running correlations for all series...')
print('')

for series in SERIES_NAMES:
    print('--- ' + series + ' ---')
    scored_rows = load_csv(os.path.join(SCORED_DIR, series + '_scored.csv'))
    monthly_rows = load_csv(os.path.join(MONTHLY_DIR, series + '_monthly.csv'))

    if scored_rows:
        all_results.extend(correlate_per_comment(scored_rows, series))
    if monthly_rows:
        all_results.extend(correlate_monthly(monthly_rows, series))
    print('')

# --- apply Benjamini-Hochberg correction across all p-values together ---
print('Applying Benjamini-Hochberg correction across ' + str(len(all_results)) + ' tests...')
raw_p_values = [r['p_value'] for r in all_results]
adjusted_p_values = benjamini_hochberg(raw_p_values)

significant_before = 0
significant_after = 0

for r, adj_p in zip(all_results, adjusted_p_values):
    r['p_value_adjusted'] = round(adj_p, 4)
    r['significant_before_correction'] = r['p_value'] < ALPHA
    r['significant_after_correction'] = adj_p < ALPHA
    if r['significant_before_correction']:
        significant_before += 1
    if r['significant_after_correction']:
        significant_after += 1

print('')
print('Summary: ' + str(significant_before) + ' of ' + str(len(all_results))
      + ' tests significant at p<0.05 before correction.')
print('Summary: ' + str(significant_after) + ' of ' + str(len(all_results))
      + ' tests still significant after Benjamini-Hochberg correction.')
print('')

if significant_after > 0:
    print('Tests that remain significant after correction:')
    for r in all_results:
        if r['significant_after_correction']:
            print('  ' + r['series'] + ' [' + r['level'] + '] ' + r['sentiment_field']
                  + ' vs ' + r['engagement_field'] + ': rho=' + str(r['spearman_rho'])
                  + ', adjusted p=' + str(r['p_value_adjusted']))
else:
    print('No tests remain significant after correction.')

write_csv(os.path.join(OUTPUT_DIR, 'engagement_correlation.csv'), all_results)
print('')
print('Done. Correlation results (with correction) written to ' + OUTPUT_DIR)