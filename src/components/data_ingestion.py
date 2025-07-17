# Import all the required libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
# Assuming you have a main execution block like this:
if __name__ == '__main__':
    # Initialize Data Ingestion
    obj = DataIngestion()
    # Execute data ingestion, it returns the paths to the raw train/test CSVs
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Initialize Data Transformation
    data_transformation = DataTransformation()
    # Execute data transformation, and CAPTURE its return values
    # These are the transformed NumPy arrays and the path to the preprocessor object
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Initialize Model Trainer
    model_trainer = ModelTrainer()
    # Now, pass the captured train_arr, test_arr, and preprocessor_path to model_trainer
    r2_score = model_trainer.initiate_model_training(train_arr, test_arr, preprocessor_path) # Assuming it returns r2_score
    print(f"Final Model R2 Score: {r2_score}")