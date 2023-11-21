import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj): ###serialization
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj: ##wb writing binary
            pickle.dump(obj, file_obj)#Serializes the Python object obj and writes it to the file using the pickle.dump method. Serialization converts the object into a byte stream suitable for storage or transmission.

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}##The report dictionary is used to store the evaluation results for each model.
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score ## test_model_score is value and list(model.key )=key here

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path): ## deserialization
    try:
        with open(file_path,'rb') as file_obj: ## rb = read binary
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)

    