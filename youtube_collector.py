import os
import time
import pandas as pd
from googleapiclient.discovery import build

# Your API key from Google Cloud Console
API_KEY = 'AIzaSyBvf9xun-n-8tc-ZGfIcvYl21QUBOwZnv8 '

# label for each series collected
SERIES_NAME = 'baby_reindeer'

# Two video IDs per series — 1500 comments each = 3000 per series
VIDEO_IDS = [
    'eafm1gB6SCM',
    'KRoD0pX-AUM',
]

# Path to Coventry OneDrive folder where data will be saved
OUTPUT_PATH = r'C:\Users\rejoi\OneDrive - Coventry University\dissertation-sentiment-analysis\dissertation_data'

# 1500 per video x 2 videos = 3000 per series
MAX_COMMENTS_PER_VIDEO = 1500

youtube = build('youtube', 'v3', developerKey=API_KEY)


def get_comments(video_id, max_results=1500):
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            textFormat='plainText',
            order='time'  # chronological order for temporal analysis
        )
        response = request.execute()

        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'video_id': video_id,
                'comment_id': item['snippet']['topLevelComment']['id'],
                'comment_text': snippet['textDisplay'],
                'author': snippet['authorDisplayName'],
                'published_at': snippet['publishedAt'],  # timestamp for temporal analysis
                'like_count': snippet['likeCount'],
                'reply_count': item['snippet']['totalReplyCount'],
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

        time.sleep(0.5)

    return comments


def get_video_metadata(video_id):
    request = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    )
    response = request.execute()
    if not response['items']:
        return {}
    item = response['items'][0]
    return {
        'video_id': video_id,
        'video_title': item['snippet']['title'],
        'video_published_at': item['snippet']['publishedAt'],
        'view_count': item['statistics'].get('viewCount', None),
        'comment_count': item['statistics'].get('commentCount', None),
    }


# --- collect comments and metadata for all videos ---
all_comments = []
video_meta_list = []

for vid_id in VIDEO_IDS:
    print('Collecting video: ' + vid_id)

    meta = get_video_metadata(vid_id)
    video_meta_list.append(meta)
    print('Title: ' + meta.get('video_title', 'N/A'))

    comments = get_comments(vid_id, max_results=MAX_COMMENTS_PER_VIDEO)
    all_comments.extend(comments)
    print('Comments collected: ' + str(len(comments)))

    time.sleep(1)

# --- save to CSV on OneDrive ---
os.makedirs(OUTPUT_PATH, exist_ok=True)

comments_df = pd.DataFrame(all_comments)
meta_df = pd.DataFrame(video_meta_list)

comments_file = os.path.join(OUTPUT_PATH, SERIES_NAME + '_comments_raw.csv')
meta_file = os.path.join(OUTPUT_PATH, SERIES_NAME + '_video_metadata.csv')

comments_df.to_csv(comments_file, index=False, encoding='utf-8-sig')
meta_df.to_csv(meta_file, index=False, encoding='utf-8-sig')

print('Done. ' + str(len(all_comments)) + ' comments saved to ' + comments_file)