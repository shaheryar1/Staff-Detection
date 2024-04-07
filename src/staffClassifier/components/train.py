from staffClassifier.components.dataset import StaffDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from staffClassifier.entity import TrainConfig, ModelConfig
from staffClassifier.components.dataset import prepare_dataset, data_transforms
from staffClassifier.components.models import ImageClassifier
from staffClassifier import logger
import mlflow.pytorch
from mlflow import MlflowClient

from staffClassifier.config_manager.configuration import ConfigurationManager


class Trainer:
    def __init__(self, model_config: ModelConfig , train_config: TrainConfig) -> None:
        
        self.train_config = train_config
        self.model_config = model_config

    def prepare_dataloader(self, train_df : pd.DataFrame,data_transform, batch_size = 32,):
        return DataLoader(StaffDataset(train_df, data_transform), batch_size=batch_size, shuffle=True)
    
    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Selected device {device.type} for training')
        logger.info(f'Training with configs')
        logger.info(self.train_config)
        train_df, val_df = prepare_dataset(self.train_config.dataset_path)
        tl = self.prepare_dataloader(train_df,data_transforms['train'],  self.train_config.batch_size)
        vl = self.prepare_dataloader(val_df, data_transforms['val'], self.train_config.batch_size)
        logger.info(f'Prepared dataloader')

        model = ImageClassifier(model_config = self.model_config, 
                                train_config= self.train_config)
        logger.info(f'Loaded model : {self.model_config.name} for training')

        trainer = pl.Trainer(max_epochs=self.train_config.epochs,
                      accelerator=device.type,default_root_dir=self.train_config.model_save_path)
        logger.info('Starting training')
        mlflow.pytorch.autolog(checkpoint=True)
        with mlflow.start_run() as run:
            trainer.fit(model=model,
                    train_dataloaders=tl,
                    val_dataloaders=vl)
            
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))



def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")

if __name__ == '__main__':

    cm = ConfigurationManager()
    t = Trainer(cm.get_model_config(), cm.get_train_config())
    t.train()   
