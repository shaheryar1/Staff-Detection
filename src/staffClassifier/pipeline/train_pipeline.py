from staffClassifier.config_manager.configuration import ConfigurationManager
from staffClassifier import logger
from staffClassifier.components.train import Trainer

class TrainPipeline:

    def __init__(self) -> None:
        pass
    
    def main(self):
        cm = ConfigurationManager()
        t = Trainer(cm.get_model_config(), cm.get_train_config())
        t.train()   