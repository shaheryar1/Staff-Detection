from staffClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from staffClassifier.utils.common import *
from dataclasses import dataclass
from pathlib import Path    

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH ):
        
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_file_path)

        print(self.config)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config