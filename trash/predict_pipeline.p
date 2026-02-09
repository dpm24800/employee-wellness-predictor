import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

'''
This method:
    Loads the trained ML model
    Loads the saved preprocessor
    Applies the same preprocessing used during training
    Runs prediction on new input data
    Returns the predicted result
    Handles errors cleanly using CustomException

In short: Raw input DataFrame → Scaled data → Model prediction
'''
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path = model_path) # 
            preprocessor = load_object(file_path = preprocessor_path) #
            
            # Apply the trained preprocessing pipeline to input features and returns the transformed (scaled/encoded) data ready for the ML model.
            data_scaled=preprocessor.transform(features)
            
            # Generate the model’s predicted target values for the given (already preprocessed) input data.
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

"""
# CustomData is a data container + formatter.
This class is used to collect user input data and convert 
it into a Pandas DataFrame for model prediction.
"""
class CustomData:
    """
    STEP 1: Initialize the object with user-provided values
    (usually coming from an HTML form or API request)
    """
    def __init__(self, 
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int
                ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    
    def get_data_as_data_frame(self):
        """
        STEP 2: Convert stored user inputs into a Pandas DataFrame
        so it can be passed to the ML pipeline
        """
        try:
            # STEP 3: Create a dictionary where
            # keys = feature names (same as training data)
            # values = list of values (even for one record)
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            # STEP 4: Convert dictionary into Pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # STEP 5: Catch any error and raise custom exception
            # for better debugging and logging
            raise CustomException (e, sys)