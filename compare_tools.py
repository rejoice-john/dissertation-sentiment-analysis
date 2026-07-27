'''
Tool comparison pipeline.

Compares VADER, TextBlob, and AFINN two ways: percentage agreement and
pairwise Cohen's Kappa on the categorical labels (positive/negative/
neutral) each tool assigns, and pairwise Spearman correlation on the raw
continuous scores themselves. The label-based comparison and the
raw-score comparison answer different questions -- do the tools classify
comments the same way, versus do their underlying scores move together --
so both are reported. The raw-score p-values are corrected for multiple
testing (Benjamini-Hochberg) since several tests are run across series.
'''

import os
import csv
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

SCORED_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\comparison'

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

ALPHA = 0.05


def load_series(series):
    # Reads one series' scored CSV
    input_path = os.path.join(SCORED_DIR, series + '_scored.csv')
    if not os.path.exists(input_path):
        print('No scored file found for ' + series + ', skipping.')
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def benjamini_hochberg(p_values):
    # Adjusts a list of p-values for multiple testing (false discovery rate)
    m = len(p_values)
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda pair: pair[1])

    adjusted = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        original_index, p = indexed[rank - 1]
        candidate = p * m / rank
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min

    return adjusted


def compare_labels(rows):
    # Calculates overall percentage agreement and pairwise Cohen's Kappa
    # between each pair of tools' classifications
    vader_labels = [r['vader_label'] for r in rows]
    textblob_labels = [r['textblob_label'] for r in rows]
    afinn_labels = [r['afinn_label'] for r in rows]

    total = len(rows)
    all_three_agree = sum(
        1 for v, t, a in zip(vader_labels, textblob_labels, afinn_labels) if v == t == a
    )
    agreement_pct = round(all_three_agree / total * 100, 2) if total else 0

    vader_textblob_kappa = round(cohen_kappa_score(vader_labels, textblob_labels), 4)
    vader_afinn_kappa = round(cohen_kappa_score(vader_labels, afinn_labels), 4)
    textblob_afinn_kappa = round(cohen_kappa_score(textblob_labels, afinn_labels), 4)

    return {
        'total_comments': total,
        'all_three_agree_pct': agreement_pct,
        'vader_textblob_kappa': vader_textblob_kappa,
        'vader_afinn_kappa': vader_afinn_kappa,
        'textblob_afinn_kappa': textblob_afinn_kappa,
    }


def correlate_scores(rows, series):
    # Correlates the three tools' raw continuous scores against each
    # other (not their categorical labels), using Spearman rank
    # correlation so AFINN's unbounded scale doesn't distort the result
    vader = [float(r['vader_compound']) for r in rows]
    textblob = [float(r['textblob_polarity']) for r in rows]
    afinn = [float(r['afinn_score']) for r in rows]

    pairs = [
        ('vader_textblob', vader, textblob),
        ('vader_afinn', vader, afinn),
        ('textblob_afinn', textblob, afinn),
    ]

    results = []
    for pair_name, a, b in pairs:
        rho, p_value = spearmanr(a, b)
        print('  [score correlation] ' + series + ' / ' + pair_name + ': '
              + 'rho=' + str(round(rho, 4)) + ', p=' + str(round(p_value, 4)))
        results.append({
            'series': series,
            'tool_pair': pair_name,
            'n': len(rows),
            'spearman_rho': round(rho, 4),
            'p_value': round(p_value, 4),
        })
    return results


def write_csv(path, rows):
    # Writes rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- compare tool agreement (labels) for each series and overall ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
label_results = []
score_results = []
all_rows = []

print('Comparing tools for all series...')
print('')

for series in SERIES_NAMES:
    print('--- ' + series + ' ---')
    rows = load_series(series)
    if not rows:
        continue
    all_rows.extend(rows)

    label_summary = compare_labels(rows)
    label_summary['series'] = series
    label_results.append(label_summary)
    print('  [labels] agreement=' + str(label_summary['all_three_agree_pct']) + '%, '
          + 'VADER-TextBlob kappa=' + str(label_summary['vader_textblob_kappa']) + ', '
          + 'VADER-AFINN kappa=' + str(label_summary['vader_afinn_kappa']) + ', '
          + 'TextBlob-AFINN kappa=' + str(label_summary['textblob_afinn_kappa']))

    score_results.extend(correlate_scores(rows, series))
    print('')

print('--- ALL_SERIES_COMBINED ---')
overall_label_summary = compare_labels(all_rows)
overall_label_summary['series'] = 'ALL_SERIES_COMBINED'
label_results.append(overall_label_summary)
score_results.extend(correlate_scores(all_rows, 'ALL_SERIES_COMBINED'))
print('')

# --- apply Benjamini-Hochberg correction to the score correlation p-values ---
print('Applying Benjamini-Hochberg correction across ' + str(len(score_results)) + ' score correlation tests...')
raw_p_values = [r['p_value'] for r in score_results]
adjusted_p_values = benjamini_hochberg(raw_p_values)

significant_before = 0
significant_after = 0

for r, adj_p in zip(score_results, adjusted_p_values):
    r['p_value_adjusted'] = round(adj_p, 4)
    r['significant_before_correction'] = r['p_value'] < ALPHA
    r['significant_after_correction'] = adj_p < ALPHA
    if r['significant_before_correction']:
        significant_before += 1
    if r['significant_after_correction']:
        significant_after += 1

print('')
print('Summary: ' + str(significant_before) + ' of ' + str(len(score_results))
      + ' score correlation tests significant at p<0.05 before correction.')
print('Summary: ' + str(significant_after) + ' of ' + str(len(score_results))
      + ' still significant after Benjamini-Hochberg correction.')
print('')

# Reorders label results so 'series' appears first
ordered_label_results = []
for r in label_results:
    ordered = {'series': r['series']}
    ordered.update({k: v for k, v in r.items() if k != 'series'})
    ordered_label_results.append(ordered)

write_csv(os.path.join(OUTPUT_DIR, 'tool_comparison.csv'), ordered_label_results)
write_csv(os.path.join(OUTPUT_DIR, 'tool_score_correlation.csv'), score_results)
print('Done. Label comparison and score correlation results written to ' + OUTPUT_DIR)