import sys
sys.path.append(r'E:\ineuron_2023\practice\practice_endtoend_oct_proj')




from src.DimondPricePrediction.components.data_ingestion import DataIngestion

#from src.DimondPricePrediction.components.data_transformation import DataTransformation

#from src.DimondPricePrediction.components.model_trainer import ModelTrainer

#from src.DimondPricePrediction.components.model_evaluation import ModelEvaluation


import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
import pandas as pd

obj=DataIngestion()

obj.initiate_data_ingestion()
