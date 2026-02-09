import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component.")
        
        try:
            # CRITICAL FIX: Resolve path relative to project root
            project_root = Path(__file__).resolve().parents[2]  # Goes up: components/ -> src/ -> project_root
            data_path = project_root / "notebook" / "data" / "WorkFromHomeBurnout.csv"
            
            # Debugging aid (optional but helpful)
            logging.info(f"Project root: {project_root}")
            logging.info(f"Looking for dataset at: {data_path}")
            
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found at: {data_path}\n"
                    f"Current working directory: {Path.cwd()}\n"
                    f"Please ensure the file exists at: notebook/data/WorkFromHomeBurnout.csv"
                )

            df = pd.read_csv(str(data_path))  # Convert Path to string for pandas
            logging.info("Read the dataset as dataframe.")
            
            # Renamed column names
            df.columns = df.columns.str.replace(' ', '_', regex=True).str.replace('/', '_', regex=True)
            logging.info(f"Renamed columns: {df.columns.tolist()}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train-test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# Standalone ingestion test only
if __name__ == "__main__":
    # Standalone test of data ingestion component only
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    # print(f"\nData ingestion standalone test completed!")
    # print(f"   Train data: {train_path}")
    # print(f"   Test data: {test_path}\n")

    # data_transformation = DataTransformation()
    # train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_path, test_path)

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))