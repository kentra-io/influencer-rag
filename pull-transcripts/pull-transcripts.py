import json
import os
import re

import requests
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

import config

def getUploadsPlaylistId(channel_id, youtube):
    request = youtube.channels().list(part='contentDetails', id=channel_id)
    response = request.execute()
    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return uploads_playlist_id


def scraping_get_channel_id_from_handle(handle: str):
    if handle.find('@') == -1:
        handle = '@' + handle

    url = 'https://www.youtube.com/' + handle
    resp = requests.get(url)

    if resp.status_code == 200:
        found = re.findall('"channelId":"([^"]*)"', resp.text)

        return found[0]
    else:
        return False

def get_all_transcripts(channelHandle, api_key, file_path, language='en'):
    channel_id = scraping_get_channel_id_from_handle(channelHandle)
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Start the JSON array
    with open(file_path, 'w') as file:
        file.write('[')

    first_video = True  # Flag to track if the video is the first in the list

    # Get all video IDs from the channel
    next_page_token = None
    counter = 0
    while True:
        request = youtube.playlistItems().list(
            part='snippet',
            playlistId=getUploadsPlaylistId(channel_id, youtube),
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_title = item['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            channel_name = item['snippet']['channelTitle']

            counter += 1
            count = response.get("pageInfo").get("totalResults")
            print(f"Processing video {counter} of {count} for channel {channelHandle}")

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript([language])
                transcript_text = ' '.join([segment['text'] for segment in transcript.fetch()])

                video_data = {
                    'title': video_title,
                    'url': video_url,
                    'channel': channel_name,
                    'transcript': transcript_text
                }

                # Append the video data to the file
                with open(file_path, 'a') as file:
                    if not first_video:
                        file.write(',')
                    json.dump(video_data, file)
                    first_video = False

            except NoTranscriptFound:
                print(f"No transcript found for video ID: {video_id}")
            except Exception as e:
                print(f"Error retrieving transcript for video ID {video_id}: {e}")

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Close the JSON array
    with open(file_path, 'a') as file:
        file.write(']')

def main():
    for channel in config.channelYoutubeHandles:
        parent_file_path = "../transcripts/"

        if not os.path.exists(parent_file_path):
            os.makedirs(parent_file_path)

        file_path = f"{parent_file_path}/{channel}_transcripts.json"
        get_all_transcripts(channel, config.API_KEY, file_path)

main()
