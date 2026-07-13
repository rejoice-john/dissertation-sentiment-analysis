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
OUTPUT_DIR = 'data'
RETRY_LIMIT = 5
WINDOW_DAYS = 365
PROGRESS_EVERY = 25   # print a progress line every 25 pages while paging

# Each series is a list of video specs, targeting 5000 comments per series
# where the video(s) have enough within the 12-month window.
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
    digest = hashlib.sha256(author_name.encode('utf-8')).hexdigest()
    return digest[:16]

def parse_time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))