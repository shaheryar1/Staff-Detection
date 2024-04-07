from staffClassifier.pipeline.data_pipeline import DataIngestionPipeline
from staffClassifier import logger
from staffClassifier.pipeline.train_pipeline import TrainPipeline

try:
    logger.info('====== Running Data Pipeline ========')
    p1 = DataIngestionPipeline()
    p1.run()
except Exception as e:
    logger.error(e)


try:
    logger.info('====== Running Train Pipeline ========')
    t = TrainPipeline()
    t.run()
except Exception as e:
    logger.error(e)