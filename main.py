from staffClassifier.components.data_ingestion import DataIngestion
from staffClassifier.config_manager.configuration import ConfigurationManager



conf = ConfigurationManager()
data_ingestion= DataIngestion((conf.get_data_ingestion_config()))
data_ingestion.extract_zip_file()