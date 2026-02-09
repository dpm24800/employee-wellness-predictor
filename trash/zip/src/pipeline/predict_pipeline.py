import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Apply the same preprocessing used during training
            data_scaled = preprocessor.transform(features)

            # Predict burnout_score
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class collects user input and converts it
    into a DataFrame for burnout score prediction.
    """

    def __init__(
        self,
        day_type: int,
        work_hours: float,
        screen_time_hours: float,
        meetings_count: int,
        breaks_taken: int,
        after_hours_work: float,
        sleep_hours: float,
        task_completion_rate: float
    ):

        self.day_type = day_type
        self.work_hours = work_hours
        self.screen_time_hours = screen_time_hours
        self.meetings_count = meetings_count
        self.breaks_taken = breaks_taken
        self.after_hours_work = after_hours_work
        self.sleep_hours = sleep_hours
        self.task_completion_rate = task_completion_rate

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "day_type": [self.day_type],
                "work_hours": [self.work_hours],
                "screen_time_hours": [self.screen_time_hours],
                "meetings_count": [self.meetings_count],
                "breaks_taken": [self.breaks_taken],
                "after_hours_work": [self.after_hours_work],
                "sleep_hours": [self.sleep_hours],
                "task_completion_rate": [self.task_completion_rate]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
