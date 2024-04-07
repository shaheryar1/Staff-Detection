from staffClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from staffClassifier.utils.common import *
from dataclasses import dataclass
from pathlib import Path    
from staffClassifier.entity import *



class ConfigurationManager:
    def __init__(self, config_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH ):
        
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_file_path)

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
    
    def get_model_config(self)->ModelConfig:
       
        params = self.params.model
        
        model_config = ModelConfig(
            name = params.name,
            input_size = params.input_size,
            num_classes =  params.num_classes,
            pretrained =  params.pretrained,
            checkpoint = params.checkpoint
        )

        return model_config

    def get_train_config(self)->TrainConfig:
        params = self.params.train

        train_config = TrainConfig(
            learning_rate=params.learning_rate,
            epochs=params.epochs,
            batch_size= params.batch_size,
            model_save_path= params.model_save_path,
            dataset_path= params.dataset_path
        )
        return train_config
