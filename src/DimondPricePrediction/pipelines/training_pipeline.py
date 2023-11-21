import sys
sys.path.append(r'E:\ineuron_2023\practice\practice_endtoend_oct_proj')



from src.DimondPricePrediction.components.data_ingestion import DataIngestion

from src.DimondPricePrediction.components.data_transformation import DataTransformation

from src.DimondPricePrediction.components.model_trainer import ModelTrainer

#from src.DimondPricePrediction.components.model_evaluation import ModelEvaluation


import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
import pandas as pd

obj=DataIngestion()

train_data_path,test_data_path=obj.initiate_data_ingestion() ## method

data_transformation=DataTransformation() ## obj of datatransf

train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)
##initialize_data_transformation method of the data_transformation object (which is an instance of the DataTransformation class). This
#  method reads, preprocesses, and transforms the training and testing datasets, and it returns the final arrays (train_arr and test_arr) that are
#  ready to be used for training and testing machine learning models.
#initialize_data_transformation(self,train_path,test_path): come from data transformation file

model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_training(train_arr,test_arr)## methid initate_model_training taking 2 arguments train arr, test arr
##initate_model_training(self,train_array,test_array):from model trainer file

#model_trainer()this class seems to be designed for training machine learning models.