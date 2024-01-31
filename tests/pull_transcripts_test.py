import pytest

import json
import logging
import os
import unittest
from unittest.mock import patch

from app._1_pull_transcripts import get_all_transcripts  # Import your method
from app.common import file_utils

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True


class TestGetAllTranscripts(unittest.TestCase):

    def setUp(self):
        # File path setup
        self.file_path = 'tests/test_data/test_transcripts.json'
        # Ensure the file is deleted if it exists
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        # Ensure the directory exists
        file_utils.createFolderIfNotExists(self.file_path)

    def test_get_all_transcripts(self):
        with open('tests/resources/pull-transcripts-test/playlist.json', 'r') as file:
            mock_playlist_items_response = json.loads(file.read())

        with patch('app._1_pull_transcripts.getPlaylistItems', return_value=mock_playlist_items_response) as mock_get:
            logger.debug("log test")
            channel_id = 'UCDXTQ8nWmx_EhZ2v-kp7QxA'
            get_all_transcripts(channel_id, "mock_api_key", self.file_path)

        self.assertTranscriptGotCreated()

    def assertTranscriptGotCreated(self):
        with open(self.file_path, 'r') as output_file:
            output_data = json.load(output_file)

        with open('tests/resources/pull-transcripts-test/expected_transcript_without_transcript_field.json',
                  'r') as expected_file:
            expected_data = json.load(expected_file)

        self.assertEqual(len(output_data), len(expected_data), "Output file should contain the same number of entries.")

        for actual, expected in zip(output_data, expected_data):
            for key in expected.keys():
                self.assertEqual(actual[key], expected[key], f"{key} does not match.")
            self.assertTrue(len(actual['transcript']) > 200, "Transcript is not long enough, looks suspicious.")


if __name__ == '__main__':
    unittest.main()
