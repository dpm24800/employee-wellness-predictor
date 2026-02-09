import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):
        """
        Builds the same ColumnTransformer used in the notebook
        """

        try:
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("OneHotEncoder", categorical_transformer, categorical_features),
                    ("StandardScaler", numeric_transformer, numerical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the same manual preprocessing steps
        used in the notebook.
        """

        # Drop columns
        drop_cols = ["user_id", "burnout_risk"]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=col)

        # Ordinal encoding for day_type
        if "day_type" in df.columns and df["day_type"].dtype == object:
            df["day_type"] = df["day_type"].map(
                {"Weekend": 0, "Weekday": 1}
            )

        # Clip rate type columns
        if "task_completion_rate" in df.columns:
            df["task_completion_rate"] = df["task_completion_rate"].clip(
                0, 100)

        if "burnout_score" in df.columns:
            df["burnout_score"] = df["burnout_score"].clip(0, 100)

        return df

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # Apply notebook style preprocessing
            train_df = self._basic_preprocessing(train_df)
            test_df = self._basic_preprocessing(test_df)

            target_column_name = "burnout_score"

            # Split input and target
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            # Same feature selection logic as notebook
            numerical_features = X_train.select_dtypes(
                exclude="object"
            ).columns.tolist()

            categorical_features = X_train.select_dtypes(
                include="object"
            ).columns.tolist()

            logging.info(f"Numerical features : {numerical_features}")
            logging.info(f"Categorical features : {categorical_features}")

            preprocessing_obj = self.get_data_transformer_object(
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )

            logging.info("Fitting preprocessor on training data")

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                X_train_transformed,
                np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_transformed,
                np.array(y_test)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
