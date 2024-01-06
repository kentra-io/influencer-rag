import os


def createFolderIfNotExists(parent_file_path):
    if not os.path.exists(parent_file_path):
        os.makedirs(parent_file_path)

