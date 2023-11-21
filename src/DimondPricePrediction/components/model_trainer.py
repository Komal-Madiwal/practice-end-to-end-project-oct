
import pandas as pd
import numpy as np
import os
import sys
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception
from dataclasses import dataclass
from src.DimondPricePrediction.utils.utils import save_object
from src.DimondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl') ##saving model.pkl file in artifacts folder
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],### x train = all rows and columns except last 
                train_array[:,-1],## y train = all rows and last col
                test_array[:,:-1],## x test
                test_array[:,-1]## y test
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            ##evaluate_model function is defined in the utils module.
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models) ## This is a variable that will store the output of the 
            #evaluate_model function.dict: It specifies the expected type of the variable. In this context, it means that model_report is expected to be a dictionary.
            ##models: A dictionary containing regression models.
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))##This extracts the values (R-squared scores) from the model_report dictionary

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]##list(model_report.values()).index(best_model_score) calculates the index of the highest R-squared score in the list of R-squared scores.
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )## save at artifacts folder
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

        
    
