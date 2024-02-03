import json
import os
import unittest
from unittest.mock import patch

from app._1_pull_transcripts import get_all_transcripts
from app.common import file_utils


class TestGetAllTranscripts(unittest.TestCase):

    def setUp(self):
        self.file_path = 'tests/test_data/test_transcripts.json'
        # Ensure the file is deleted if it exists
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        # Ensure the directory exists
        file_utils.createFolderIfNotExists(self.file_path)

    def test_get_all_transcripts(self):
        with open('tests/resources/pull-transcripts-test/playlist.json', 'r') as file:
            mock_playlist_items_response = json.loads(file.read())

        with patch('app._1_pull_transcripts.get_playlist_items', return_value=mock_playlist_items_response):
            channel_id = 'UCDXTQ8nWmx_EhZ2v-kp7QxA'
            get_all_transcripts(channel_id, self.file_path)

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
            self.assertGreater(len(actual['transcript']), 200, "Transcript is not long enough, looks suspicious.")


if __name__ == '__main__':
    unittest.main()
