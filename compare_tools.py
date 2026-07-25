'''
Tool comparison pipeline.

Compares the sentiment labels assigned by VADER, TextBlob, and AFINN to
measure how often the three tools agree, and where they diverge from
each other pairwise, addressing objective 4.
'''

import os
import csv
from sklearn.metrics import cohen_kappa_score

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\comparison'

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]


def load_series(series):
    # Reads one series' scored CSV
    input_path = os.path.join(INPUT_DIR, series + '_scored.csv')
    if not os.path.exists(input_path):
        print('No scored file found for ' + series + ', skipping.')
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def compare_labels(rows):
    # Calculates overall percentage agreement and pairwise Cohen's Kappa
    # between each pair of tools
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


def write_csv(path, rows):
    # Writes the comparison results out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- compare tool agreement for each series and overall ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = []
all_rows = []

for series in SERIES_NAMES:
    rows = load_series(series)
    if not rows:
        continue
    all_rows.extend(rows)
    summary = compare_labels(rows)
    summary['series'] = series
    results.append(summary)
    print('Compared ' + str(summary['total_comments']) + ' comments for ' + series)

overall_summary = compare_labels(all_rows)
overall_summary['series'] = 'ALL_SERIES_COMBINED'
results.append(overall_summary)

# Reorders so 'series' appears as the first column in the CSV
ordered_results = []
for r in results:
    ordered = {'series': r['series']}
    ordered.update({k: v for k, v in r.items() if k != 'series'})
    ordered_results.append(ordered)

write_csv(os.path.join(OUTPUT_DIR, 'tool_comparison.csv'), ordered_results)
print('Done. Tool comparison written to ' + OUTPUT_DIR)