import json
import logging
import os

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound

from app import config
from app.common import file_utils

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
logger = logging.getLogger(__name__)


def get_uploads_playlist_id(channel_id, youtube):
    request = youtube.channels().list(part='contentDetails', id=channel_id)
    response = request.execute()
    uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    return uploads_playlist_id


def get_all_transcripts(channel_id, file_path, language='en'):
    youtube = build('youtube', 'v3', developerKey=GOOGLE_API_KEY)

    # Start the JSON array
    with open(file_path, 'w') as file:
        file.write('[')

    first_video = True  # Flag to track if the video is the first in the list

    # Get all video IDs from the channel
    next_page_token = None
    counter = 0
    while True:
        playlistItemsResponse = getPlaylistItems(youtube, next_page_token, channel_id)

        for item in playlistItemsResponse['items']:
            video_id = item['snippet']['resourceId']['videoId']
            video_title = item['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            channel_name = item['snippet']['channelTitle']

            counter += 1
            count = playlistItemsResponse.get("pageInfo").get("totalResults")
            logger.info(f"Processing video {counter} of {count}")

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
                logger.error(f"No transcript found for video ID: {video_id}")
            except Exception as e:
                logger.error(f"Error retrieving transcript for video ID {video_id}: {e}")

        next_page_token = playlistItemsResponse.get('nextPageToken')
        if not next_page_token:
            break

    # Close the JSON array
    with open(file_path, 'a') as file:
        file.write(']')


def getPlaylistItems(youtube, next_page_token, channel_id):
    request = youtube.playlistItems().list(
        part='snippet',
        playlistId=get_uploads_playlist_id(channel_id, youtube),
        maxResults=50,
        pageToken=next_page_token
    )
    return request.execute()


def main():
    for channel in config.channels:
        file_path = f"{config.transcripts_dir_path}/{channel.handle}_transcripts.json"
        file_utils.createFolderIfNotExists(file_path)

        if not os.path.exists(file_path):
            logger.info(f"Retrieving transcriptions for channel '{channel.handle}':")
            get_all_transcripts(channel.id, file_path)
        else:
            logger.warning(f"File '{file_path}' exists, skipping retrieval for channel '{channel.handle}'")


if __name__ == "__main__":
    main()
