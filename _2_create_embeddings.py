import json
import logging
import os
import time
from datetime import timedelta

import config
from retrieval.punctuator import punctuate
from retrieval.tiler import get_sentences, create_tiles
from utils.console_utils import bold
from vector_db import chroma_provider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

chroma = chroma_provider.get_chroma()


def process_transcript(file_path):
    logger.info(f"Processing file: {file_path}")

    with open(file_path, 'r') as file:
        data_list = json.load(file)

        for data in data_list:
            try:
                logger.info(f"Processing '{data['title']}'")

                tiling_start_time = time.time()

                punctuated_transcript = punctuate(data['transcript']).replace('\n', ' ')

                transcript_sentences = get_sentences(punctuated_transcript)

                tiles = create_tiles(transcript_sentences)

                # TODO Move to a separate method or update create_tiles method
                paragraphs = []
                current_par = ""
                for i, sentence in enumerate(transcript_sentences):
                    current_par += " " + sentence

                    if i + 1 in tiles[1:-1]:
                        paragraphs.append(current_par)
                        current_par = ""

                paragraphs.append(current_par)
                # End

                tiling_time = time.time() - tiling_start_time

                logger.info(
                    f"Created {bold(len(paragraphs))} paragraphs in {bold(str(timedelta(seconds=tiling_time)))}")

                video_metadata = {
                    "title": data['title'],
                    "channel": data['channel'],
                    "url": data['url'],
                    "file": file_path
                }

                for paragraph in paragraphs:
                    chunk_metadata = video_metadata.copy()
                    # TODO That is super not optimal and temporary solution, we should keep
                    #  paragraphs and chunks in a separate database and use IDs instead
                    #  (https://stackoverflow.com/questions/77030151/generating-and-using-chromadb-ids)
                    chunk_metadata["paragraph"] = paragraph

                    chunks = get_sentences(paragraph)

                    metadata = [chunk_metadata] * (len(chunks))

                    chroma.add_texts(texts=chunks, metadatas=metadata)

                chroma.persist()

            except Exception as e:
                logger.error(f"Unknown exception has been raised, skipping the transcript: {e}", exc_info=True)


def main():
    for channel in config.channels:
        transcript_file_path = f"{config.transcripts_dir_path}/{channel.handle}_transcripts.json"
        if os.path.exists(transcript_file_path):
            process_transcript(transcript_file_path)
            logger.info(f"Completed processing for file: {transcript_file_path}")
        else:
            logger.error(f"File: {transcript_file_path} does not exists")


if __name__ == "__main__":
    main()
