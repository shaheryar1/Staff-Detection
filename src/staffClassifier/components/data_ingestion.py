import gdown
from staffClassifier import logger
import os
import zipfile
from staffClassifier.entity import *

class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        '''
        Fetch data from the url
        '''
        try: 
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            if os.path.exists(zip_download_dir):
                logger.info('File already exists. Skipping download')
            else:

                os.makedirs("artifacts/data_ingestion", exist_ok=True)
                logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
                gdown.download(dataset_url, zip_download_dir, quiet=False)
                logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e



    def extract_zip_file(self):
        
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)