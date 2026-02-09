import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # List numerical and categrorical features
            numerical_columns = [
                'user_id', 
                'work_hours',
                'screen_time_hours',
                'meetings_count',
                'breaks_taken',
                'after_hours_work',
                'sleep_hours',
                'task_completion_rate',
                'burnout_score',
                
            ]
            categorical_features = [
                'day_type',
                'burnout_risk',
            ]

            # Set steps on num_pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Set steps on cat_pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combines different preprocessing pipelines for different column types.
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "burnout_score"  # Now valid after renaming in ingestion

            # Seperate input and target features from train and test data frame
            # - FIXED: Removed invalid 'axis=1' parameter
            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes.")

            '''
            Fit_transform on train ONLY, transform on test
            Fits the preprocessing pipeline on training data and transforms it into model-ready numerical features.
            '''
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            
            # Applies the same preprocessing rules learned from training data to the test data.
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            '''
            Horizontally combines features and target into one NumPy array.
            Appends the target column to the end of the feature matrix.
            Merges processed training features and target values into one array by adding the target as the last column.
            '''
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            '''
            Horizontally appends the test target values to the test feature matrix.
            Combines test features and test labels into one array by adding the target as the last column.
                
            Why this is done: 
            - Store test data as a single array/artifact
            - Easy to split later for evaluation
            - Keeps train & test formats identical
            '''
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the trained preprocessing pipeline so the same transformations can be reused during inference.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            '''
            Returns processed data and the path to the saved preprocessor
            Returns processed training & test data along with the path to the saved preprocessing object.
            '''
            return (
                train_arr, # preprocessed training data, with features and target combined.
                test_arr,  # preprocessed test data, also with features and target combined.
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)