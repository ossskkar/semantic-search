from datetime              import datetime
from evaluator             import Evaluator
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer
from util                  import create_data_loader
from util                  import initialize_logger
from util                  import load_config

import os

config = load_config()
logger = initialize_logger(config)
model  = SentenceTransformer(config['TRAIN']['BASE_MODEL_NAME'])

model_save_path = config['TRAIN']['MODEL_SAVE_PATH']+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

# Create data loader and loss for MultipleNegativesRankingLoss
train_dataloader_MultipleNegativesRankingLoss = create_data_loader(config, data_type='MultipleNegativesRankingLoss')
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)

# Create data loader and loss for OnlineContrastiveLoss
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
train_dataloader_ConstrativeLoss = create_data_loader(config, data_type='ConstrativeLoss')
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=config['TRAIN']['MARGIN'])

# Create evaluators
evaluator = Evaluator(config)
seq_evaluator = evaluator.sequential_evaluators()

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

train_objectives=[(train_dataloader_MultipleNegativesRankingLoss, train_loss_MultipleNegativesRankingLoss), 
                  (train_dataloader_ConstrativeLoss, train_loss_ConstrativeLoss)]

# Train the model
model.fit(train_objectives=train_objectives,
          evaluator=seq_evaluator,
          epochs=config['TRAIN']['NUM_EPOCHS'],
          warmup_steps=config['TRAIN']['WARMUP_STEPS'],
          output_path=config['TRAIN']['MODEL_SAVE_PATH']
          )