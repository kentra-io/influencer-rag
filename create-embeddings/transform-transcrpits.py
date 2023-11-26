import glob
import json
import logging

import chromadb
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define embedding dimension
embedding_dim = 384  # Example embedding dimension, adjust as per your model

# Load or create Chroma
client = chromadb.PersistentClient(path="../chroma_db")
client.get_or_create_collection("youtube_transcripts")
chroma = Chroma(
    client=client,
    collection_name="youtube_transcripts",
    embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="../chroma_db"
)


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
for file_path in glob.glob('../transcripts/*'):
    process_transcript(file_path)
    logger.info(f"Completed processing for file: {file_path}")
