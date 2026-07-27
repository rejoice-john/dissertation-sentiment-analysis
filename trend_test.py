'''
Temporal sentiment trend test pipeline.

Tests whether each series' sentiment shows a significant monotonic trend
across its twelve-month post-release window, addressing objective 3.
Uses Spearman correlation between month number and each tool's average
monthly sentiment score, rather than linear regression, since twelve
points per series is a thin basis for regression's normality and
linearity assumptions; a rank-based trend test only assumes the trend is
monotonic, not that it is linear.

Runs one test per series per tool (5 series x 3 tools = 15 tests), and
applies a Benjamini-Hochberg correction across all fifteen together,
following the same approach used in compare_tools.py and
correlate_engagement.py.

Also produces one plot per series, showing all three tools' average
monthly sentiment score across the twelve-month window, so the trend
(or lack of one) can be seen directly alongside the statistical result.
'''

import os
import csv
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\aggregated'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\trend'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]

# Used for readable plot titles rather than the raw file-name-style series keys
SERIES_LABELS = {
    'baby_reindeer': 'Baby Reindeer',
    'the_boys_s4': 'The Boys S4',
    'house_of_the_dragon_s2': 'House of the Dragon S2',
    'fallout_s1': 'Fallout S1',
    'emily_in_paris_s4': 'Emily in Paris S4',
}

TOOLS = ['vader', 'textblob', 'afinn']
TOOL_COLOURS = {'vader': '#1f77b4', 'textblob': '#ff7f0e', 'afinn': '#2ca02c'}
ALPHA = 0.05   # significance threshold used both before and after correction


def load_csv(path):
    # Reads a CSV file into a list of dictionaries
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def benjamini_hochberg(p_values):
    # Adjusts a list of p-values for multiple testing (false discovery rate),
    # same implementation used in compare_tools.py and correlate_engagement.py
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


def test_series_trend(series, rows):
    # Runs one Spearman trend test per tool for this series, correlating
    # month number (1-12) against that tool's average monthly sentiment score
    results = []
    months = [int(r['month']) for r in rows]

    for tool in TOOLS:
        avg_field = tool + '_avg_score'
        scores = [float(r.get(avg_field, 0)) for r in rows]
        rho, p_value = spearmanr(months, scores)
        print('  [trend] ' + series + ' / ' + tool + ': '
              + 'rho=' + str(round(rho, 4)) + ', p=' + str(round(p_value, 4))
              + ', n=' + str(len(rows)))
        results.append({
            'series': series,
            'tool': tool,
            'n': len(rows),
            'spearman_rho': round(rho, 4),
            'p_value': round(p_value, 4),
        })
    return results


def plot_series_trend(series, rows):
    # Plots all three tools' average monthly sentiment score across the
    # twelve-month window for one series, displayed and saved as a PNG
    months = [int(r['month']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    for tool in TOOLS:
        avg_field = tool + '_avg_score'
        scores = [float(r.get(avg_field, 0)) for r in rows]
        ax.plot(months, scores, marker='o', label=tool.upper(), color=TOOL_COLOURS[tool])

    ax.set_title('Monthly Average Sentiment: ' + SERIES_LABELS.get(series, series))
    ax.set_xlabel('Month Since Release')
    ax.set_ylabel('Average Sentiment Score')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xticks(range(1, 13))
    ax.legend()
    fig.tight_layout()

    output_path = os.path.join(PLOTS_DIR, series + '_trend.png')
    fig.savefig(output_path, dpi=150)
    plt.show()       # displays in Spyder's Plots pane
    plt.close(fig)
    print('  Plot saved: ' + output_path)


def write_csv(path, rows):
    # Writes rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- test and plot the monthly sentiment trend for each series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
all_results = []

print('Running trend tests for all series...')
print('')

for series in SERIES_NAMES:
    print('--- ' + series + ' ---')
    rows = load_csv(os.path.join(INPUT_DIR, series + '_monthly.csv'))
    if not rows:
        print('No monthly file found for ' + series + ', skipping.')
        continue

    all_results.extend(test_series_trend(series, rows))
    plot_series_trend(series, rows)
    print('')

# --- apply Benjamini-Hochberg correction across all trend tests together ---
print('Applying Benjamini-Hochberg correction across ' + str(len(all_results)) + ' trend tests...')
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
      + ' trend tests significant at p<0.05 before correction.')
print('Summary: ' + str(significant_after) + ' of ' + str(len(all_results))
      + ' still significant after Benjamini-Hochberg correction.')
print('')

if significant_after > 0:
    print('Series/tool pairs with a significant trend after correction:')
    for r in all_results:
        if r['significant_after_correction']:
            direction = 'rising' if r['spearman_rho'] > 0 else 'falling'
            print('  ' + r['series'] + ' / ' + r['tool'] + ': ' + direction
                  + ' trend, rho=' + str(r['spearman_rho']) + ', adjusted p=' + str(r['p_value_adjusted']))
else:
    print('No series/tool pairs show a significant trend after correction.')

write_csv(os.path.join(OUTPUT_DIR, 'sentiment_trend_test.csv'), all_results)
print('')
print('Done. Trend test results written to ' + OUTPUT_DIR)
print('Plots written to ' + PLOTS_DIR)