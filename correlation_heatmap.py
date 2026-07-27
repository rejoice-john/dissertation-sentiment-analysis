'''
Correlation heatmap visualisation.

Builds a 3x3 heatmap of the pairwise Spearman correlations between
VADER, TextBlob, and AFINN's raw sentiment scores, reading directly from
tool_score_correlation.csv (already produced by compare_tools.py, so no
recalculation happens here). Produces one heatmap per series plus one for
the combined dataset, matching the correlation-matrix figure style common
in the sentiment analysis literature.
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\comparison\tool_score_correlation.csv'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\comparison\heatmaps'

TOOLS = ['vader', 'textblob', 'afinn']
TOOL_LABELS = ['VADER', 'TextBlob', 'AFINN']

# Maps each tool pair to the tool_pair value written by compare_tools.py
PAIR_LOOKUP = {
    ('vader', 'textblob'): 'vader_textblob',
    ('vader', 'afinn'): 'vader_afinn',
    ('textblob', 'afinn'): 'textblob_afinn',
}


def load_csv(path):
    # Reads a CSV file into a list of dictionaries
    if not os.path.exists(path):
        print('File not found: ' + path)
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_matrix(rows, series):
    # Builds a symmetric 3x3 matrix of Spearman rho between the three
    # tools' raw scores for one series (or ALL_SERIES_COMBINED)
    series_rows = [r for r in rows if r['series'] == series]
    matrix = np.eye(3)   # diagonal = 1.0, each tool correlates perfectly with itself

    for r in series_rows:
        pair = r['tool_pair']
        rho = float(r['spearman_rho'])
        for (a, b), name in PAIR_LOOKUP.items():
            if name == pair:
                i, j = TOOLS.index(a), TOOLS.index(b)
                matrix[i, j] = rho
                matrix[j, i] = rho
    return matrix


def plot_heatmap(matrix, title, output_path):
    # Plots a 3x3 correlation matrix as a heatmap with values annotated
    # on each cell, displayed in Spyder's Plots pane and saved as a PNG
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(TOOL_LABELS)
    ax.set_yticklabels(TOOL_LABELS)

    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            text_colour = 'white' if abs(value) > 0.6 else 'black'
            ax.text(j, i, str(round(value, 3)), ha='center', va='center', color=text_colour)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Spearman rho')
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.show()       # displays in Spyder's Plots pane
    plt.close(fig)
    print('  Heatmap saved: ' + output_path)


# --- build one heatmap per series, plus one for the combined dataset ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
rows = load_csv(INPUT_PATH)

if not rows:
    print('No correlation data found. Run compare_tools.py first.')
else:
    all_series = sorted(set(r['series'] for r in rows))
    print('Building correlation heatmaps for: ' + ', '.join(all_series))
    print('')

    for series in all_series:
        matrix = build_matrix(rows, series)
        title = 'Tool Score Correlation: ' + series.replace('_', ' ').title()
        output_path = os.path.join(OUTPUT_DIR, series + '_correlation_heatmap.png')
        plot_heatmap(matrix, title, output_path)

    print('')
    print('Done. Heatmaps written to ' + OUTPUT_DIR)