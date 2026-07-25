'''
YouTube Data API v3 comment collection pipeline.
Collects a specific number of comments from each manually selected video ID
per series, restricted to a 12-month window starting from the earliest
publish date among that series' videos, running one year forward.
Comments come back newest-first from the API, so the script pages past
recent comments to reach the window.
'''

import os
import csv
import time
import hashlib
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# API key from Google Cloud Console
API_KEY = 'AIzaSyBvf9xun-n-8tc-ZGfIcvYl21QUBOwZnv8'

COMMENTS_PER_PAGE = 100
OUTPUT_DIR = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data'
RETRY_LIMIT = 5
WINDOW_DAYS = 365
PROGRESS_EVERY = 25   # print a progress line every 25 pages while paging

# Each series is a list of video specs, targeting 5000 comments per series
# where the video have enough within the 12-month window.
VIDEO_IDS = {
    'baby_reindeer': [
        {'id': 'eafm1gB6SCM', 'target': 2200},   # official trailer, 2221 total on video
        {'id': 'KRoD0pX-AUM', 'target': 2500},   # extra, 2499 total on video
        {'id': 'khstPnH89ZM', 'target': 300},    # extra, 371 total on video
    ],
    'the_boys_s4': [
        {'id': 'EzFXDvC-EwM', 'target': 5000},   # official trailer, 17563 total on video
    ],
    'house_of_the_dragon_s2': [
        {'id': 'YN2H_sKcmGw', 'target': 5000},   # official trailer, 7645 total on video
    ],
    'fallout_s1': [
        {'id': '0kQ8i2FpRDk', 'target': 5000},   # official trailer, 25912 total on video
    ],
    'emily_in_paris_s4': [
        {'id': 'aLPk8yRq9_c', 'target': 2500},   # official trailer part 1, 1641 total on video
        {'id': 'hJ1ditMDfIE', 'target': 2500},   # official trailer part 2, 1871 total on video
    ],
}


def anonymise(author_name):
    # One-way hash of the username so no raw identity is stored (ethics requirement)
    digest = hashlib.sha256(author_name.encode('utf-8')).hexdigest()
    return digest[:16]


def parse_time(value):
    # Converts YouTube's ISO 8601 timestamp (e.g. 2024-04-15T10:22:00Z) into a datetime
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def get_publish_dates(video_ids):
    # videos().list accepts up to 50 IDs per call, so this is cheap on quota
    dates = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        response = youtube.videos().list(part='snippet', id=','.join(chunk)).execute()
        for item in response.get('items', []):
            dates[item['id']] = parse_time(item['snippet']['publishedAt'])
    return dates


def flatten_thread(item, series):
    # A thread is a top-level comment plus its replies.
    # This turns that nested structure into flat rows for the CSV.
    top = item['snippet']['topLevelComment']['snippet']
    comment_id = item['snippet']['topLevelComment']['id']
    video_id = top['videoId']

    row = {
        'series': series,
        'video_id': video_id,
        'comment_id': comment_id,
        'parent_id': '',
        'author_hash': anonymise(top.get('authorDisplayName', 'unknown')),
        'text': top.get('textOriginal', ''),
        'like_count': top.get('likeCount', 0),
        'published_at': top.get('publishedAt', ''),
        'updated_at': top.get('updatedAt', ''),
        'reply_count': item['snippet'].get('totalReplyCount', 0),
    }
    rows = [row]

    if 'replies' in item:
        for reply in item['replies']['comments']:
            r = reply['snippet']
            rows.append({
                'series': series,
                'video_id': video_id,
                'comment_id': reply['id'],
                'parent_id': comment_id,   # links reply back to its parent
                'author_hash': anonymise(r.get('authorDisplayName', 'unknown')),
                'text': r.get('textOriginal', ''),
                'like_count': r.get('likeCount', 0),
                'published_at': r.get('publishedAt', ''),
                'updated_at': r.get('updatedAt', ''),
                'reply_count': 0,
            })
    return rows


def fetch_comments_for_video(video_id, series, target, window_start, window_end):
    # Comments come back newest-first, so it moves past anything more recent
    # than window_end (discarding it), collect anything inside the window,
    # and stop once we hit the target or run out of pages (the video's
    # true oldest comments).
    collected = []
    page_token = None
    page_count = 0
    has_more_pages = True   # replaces the pagination "break"

    while has_more_pages:
        retries = 0
        request_succeeded = False   # replaces the retry-loop "break"
        response = None

        # Retry transient API errors with exponential backoff
        while not request_succeeded:
            try:
                request = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=COMMENTS_PER_PAGE,
                    pageToken=page_token,
                    order='time',
                    textFormat='plainText',
                )
                response = request.execute()
                request_succeeded = True
            except HttpError as error:
                # 403/429/500/503 are usually transient
                if error.resp.status in (403, 429, 500, 503):
                    retries += 1
                    if retries > RETRY_LIMIT:
                        print('Giving up on video ' + video_id + ' after repeated errors: ' + str(error))
                        return collected
                    wait = 2 ** retries   # exponential backoff: 2s, 4s, 8s...
                    print('API error on video ' + video_id + ', retrying in ' + str(wait) + 's...')
                    time.sleep(wait)
                else:
                    # Anything else (e.g. comments disabled)
                    print('Skipping video ' + video_id + ': ' + str(error))
                    return collected

        target_reached = False
        for item in response.get('items', []):
            if not target_reached:
                rows = flatten_thread(item, series)
                for r in rows:
                    t = parse_time(r['published_at'])
                    if window_start <= t <= window_end:
                        collected.append(r)
                if len(collected) >= target:
                    target_reached = True

        if target_reached:
            return collected[:target]

        page_count += 1
        if page_count % PROGRESS_EVERY == 0:
            print('  ...video ' + video_id + ': checked ' + str(page_count) + ' pages, '
                  + str(len(collected)) + ' in-window comments so far')

        # No token means it's reached the last page of comments for this video
        page_token = response.get('nextPageToken')
        has_more_pages = page_token is not None
        if has_more_pages:
            time.sleep(0.5)

    return collected


def collect_series(series, video_specs):
    # Walks through each video ID for a series, in order, collecting
    # comments toward each video's target.
    if not video_specs:
        print('No video IDs configured for ' + series + ', skipping.')
        return []

    video_ids = [spec['id'] for spec in video_specs if spec['id']]
    if not video_ids:
        print('No valid video IDs for ' + series + ', skipping.')
        return []

    # Window = earliest publish date among this series' videos, +365 days
    publish_dates = get_publish_dates(video_ids)
    window_start = min(publish_dates.values())
    window_end = window_start + timedelta(days=WINDOW_DAYS)
    print(series + ': window is ' + str(window_start.date()) + ' to ' + str(window_end.date()))

    all_rows = []
    carry_over = 0   # unmet target from a previous video gets added here

    for i, spec in enumerate(video_specs):
        video_id = spec['id']
        if not video_id:
            print('Missing video ID at position ' + str(i) + ' in ' + series + ', skipping.')
        else:
            # If the previous video came up short, this video tries to make up
            # the difference on top of its own target.
            target = spec['target'] + carry_over
            print('Collecting from ' + series + ' / video ' + video_id + ' (target ' + str(target) + ')')
            rows = fetch_comments_for_video(video_id, series, target, window_start, window_end)
            print('Got ' + str(len(rows)) + ' comments from video ' + video_id + ' (target was ' + str(target) + ')')

            shortfall = target - len(rows)
            carry_over = shortfall if shortfall > 0 else 0
            all_rows.extend(rows)

    print('Total for ' + series + ': ' + str(len(all_rows)) + ' comments')
    return all_rows


def write_csv(path, rows):
    # Writes one series' collected rows out to a CSV file
    if not rows:
        return
    fieldnames = ['series', 'video_id', 'comment_id', 'parent_id', 'author_hash',
                  'text', 'like_count', 'published_at', 'updated_at', 'reply_count']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- collect comments for all series ---
youtube = build('youtube', 'v3', developerKey=API_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)
master_rows = []

# Collect each series separately, write its own CSV, and also keep
# a running combined list for the master file.
for series, video_specs in VIDEO_IDS.items():
    rows = collect_series(series, video_specs)
    write_csv(os.path.join(OUTPUT_DIR, series + '.csv'), rows)
    master_rows.extend(rows)

write_csv(os.path.join(OUTPUT_DIR, 'all_series_comments.csv'), master_rows)
print('Done. ' + str(len(master_rows)) + ' total comments written to ' + OUTPUT_DIR)