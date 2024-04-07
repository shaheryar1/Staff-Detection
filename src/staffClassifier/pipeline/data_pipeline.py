from staffClassifier.components.data_ingestion import DataIngestion
from staffClassifier.config_manager.configuration import ConfigurationManager
from staffClassifier import logger


class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def run(self):
        conf = ConfigurationManager()
        data_ingestion= DataIngestion((conf.get_data_ingestion_config()))
        data_ingestion.download_dataset()
        data_ingestion.extract_zip_file()
