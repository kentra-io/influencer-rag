import glob
import json
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from vector_db import chroma_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define embedding dimension
embedding_dim = 384  # Example embedding dimension, adjust as per your model

chroma = chroma_provider.setup_chroma()

def process_transcript(file_path):
    logger.info(f"Processing file: {file_path}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    with open(file_path, 'r') as file:
        data_list = json.load(file)  # Load the list of objects

        for data in data_list:  # Iterate over each object in the list
            transcript = data['transcript']  # Extract the transcript from each object

            # Split transcript into chunks
            chunks = splitter.split_text(transcript)

            for doc_number, chunk in enumerate(chunks):
                # Add embeddings to chroma collection
                chroma.add_texts(texts=[chunk], metadatas=[{"title": data['title']}])
                logger.info(f"Document {doc_number} processed and added to Chroma collection")
            chroma.persist()
            print("persisted embeddings from video: " + data['title'])


# Iterate over all files in the ../transcripts/* directory
for file_path in glob.glob(config.transcripts_dir_path + '/*'):
    process_transcript(file_path)
    logger.info(f"Completed processing for file: {file_path}")
