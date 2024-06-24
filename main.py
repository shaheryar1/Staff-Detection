from staffClassifier.pipeline.data_pipeline import DataIngestionPipeline
from staffClassifier import logger
from staffClassifier.pipeline.train_pipeline import TrainPipeline

import dagshub
dagshub.init(repo_owner='shaheryartariq909', repo_name='Staff-Detection', mlflow=True)

try:
    logger.info('====== Running Data Pipeline ========')
    p1 = DataIngestionPipeline()
    p1.main()
except Exception as e:
    logger.error(e)


try:
    logger.info('====== Running Train Pipeline ========')
    t = TrainPipeline()
    t.main()
except Exception as e:
    logger.error(e)