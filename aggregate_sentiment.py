'''
Monthly aggregation pipeline.
 
Rolls up the per-comment sentiment scores into monthly figures per series,
covering the twelve-month post-release window, so sentiment trends and
engagement metrics can be tracked over time and later correlated.
'''
 
import os
import csv
from datetime import datetime
from collections import defaultdict
 
INPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\scored'
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data\aggregated'
 
SERIES_NAMES = [
    'baby_reindeer',
    'the_boys_s4',
    'house_of_the_dragon_s2',
    'fallout_s1',
    'emily_in_paris_s4',
]
 
 
def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))
 
 
def month_number(published_at, window_start):
    # Converts a comment's timestamp into a 1-12 "months since release"
    # figure relative to the series' earliest comment, rather than a raw
    # calendar month, so different series can be compared at the same
    # point in their post-release timeline
    t = parse_time(published_at)
    months_elapsed = (t.year - window_start.year) * 12 + (t.month - window_start.month)
    n = months_elapsed + 1
    if n < 1:
        n = 1
    if n > 12:
        n = 12
    return n
 
 
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
 
 
def aggregate_series(series, rows):
    # Groups comments by month and calculates sentiment proportions and
    # engagement figures for each month
    if not rows:
        return []
 
    window_start = min(parse_time(r['published_at']) for r in rows)
    months = defaultdict(list)
    for row in rows:
        n = month_number(row['published_at'], window_start)
        months[n].append(row)
 
    results = []
    for n in range(1, 13):
        month_rows = months.get(n, [])
        if not month_rows:
            continue
 
        top_level = [r for r in month_rows if not r.get('parent_id')]
        replies = [r for r in month_rows if r.get('parent_id')]
 
        comment_volume = len(top_level)          # objective 5's "comment volume"
        reply_activity = len(replies)             # objective 5's "reply activity", kept separate
        total_likes = sum(int(r.get('like_count', 0)) for r in month_rows)
        avg_likes = total_likes / len(month_rows) if month_rows else 0
 
        summary = {
            'series': series,
            'month': n,
            'comment_volume': comment_volume,
            'reply_activity': reply_activity,
            'total_likes': total_likes,
            'avg_likes': round(avg_likes, 2),
            'avg_textblob_subjectivity': round(
                sum(float(r.get('textblob_subjectivity', 0)) for r in month_rows) / len(month_rows), 3),
        }
 
        for tool, label_field, score_field in [
            ('vader', 'vader_label', 'vader_compound'),
            ('textblob', 'textblob_label', 'textblob_polarity'),
            ('afinn', 'afinn_label', 'afinn_score'),
        ]:
            total = len(month_rows)
            positive = sum(1 for r in month_rows if r.get(label_field) == 'positive')
            negative = sum(1 for r in month_rows if r.get(label_field) == 'negative')
            neutral = sum(1 for r in month_rows if r.get(label_field) == 'neutral')
            avg_score = sum(float(r.get(score_field, 0)) for r in month_rows) / total
 
            summary[tool + '_positive_pct'] = round(positive / total * 100, 2)
            summary[tool + '_negative_pct'] = round(negative / total * 100, 2)
            summary[tool + '_neutral_pct'] = round(neutral / total * 100, 2)
            summary[tool + '_avg_score'] = round(avg_score, 4)
 
        print('  [month ' + str(n) + '] volume=' + str(comment_volume) + ', replies=' + str(reply_activity)
              + ', avg_likes=' + str(summary['avg_likes']) + ', vader=' + str(summary['vader_avg_score'])
              + ', textblob=' + str(summary['textblob_avg_score']) + ', afinn=' + str(summary['afinn_avg_score']))
 
        results.append(summary)
 
    return results
 
 
def write_csv(path, rows):
    # Writes the aggregated monthly rows out to a CSV file
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
 
# --- aggregate monthly sentiment and engagement figures for all series ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
master_rows = []
 
print('Aggregating monthly figures for all series...')
print('')
 
for series in SERIES_NAMES:
    print('--- ' + series + ' ---')
    rows = load_series(series)
    monthly = aggregate_series(series, rows)
    write_csv(os.path.join(OUTPUT_DIR, series + '_monthly.csv'), monthly)
    master_rows.extend(monthly)
    print('  Aggregated ' + str(len(monthly)) + ' months for ' + series)
    print('')
 
write_csv(os.path.join(OUTPUT_DIR, 'all_series_monthly.csv'), master_rows)
print('Done. Monthly aggregates written to ' + OUTPUT_DIR)
