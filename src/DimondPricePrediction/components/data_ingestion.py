
import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")#The os.path.join("artifacts", "raw.csv") is used to create the default file path by joining the folder name ("artifacts") with the file name ("raw.csv").
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):##n the __init__ method of the DataIngestion class, an instance of DataIngestionConfig is created and assigned 
        #to the attribute ingestion_config. This attribute is meant to hold configuration parameters related to data ingestion.
        self.ingestion_config=DataIngestionConfig()
        
    
    def initiate_data_ingestion(self):##This is a method of the DataIngestion class.
        logging.info("data ingestion started")
        
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","gemstone.csv")))
            logging.info(" i have read dataset as a df")
            
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info(" i have saved the raw dataset in artifact folder")
            
            logging.info("here i have performed train test split")
            
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)##These lines assume that self.ingestion_config.train_data_path 
            #and self.ingestion_config.test_data_path are attributes within an object (likely an instance of a class) that specify the paths where the training and testing data should be saved. 
            test_data.to_csv(self.ingestion_config.test_data_path,index=False) ## saving at artifacts folder
            
            logging.info("data ingestion part completed")
            
            return (
                 
                
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            
        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise customexception(e,sys)
    