## data_ingestion.py

- Imports:
  - os, sys, pandas, train_test_split, dataclass,
  - CustomException, logging,
  - DataTransformationConfig, DataTransformation,
  - ModelTrainerConfig, ModelTrainer
- DataIngestionConfig:
- DataIngestion:
  - __init__(self)
  - initiate_data_ingestion(self)
    - try:
      - make a df from a csv file
      - clean/rename column names
      - make artifact folder
      - export raw_data.csv from the dataframe
      - train/test split from the dataclass (test_size=0.2)
      - export train_set to train.csv
      - export test_set to test.csv
      - return train_data_path and test_data_path
    - except
      - raise CustomException

__main__:
	train_data, test_data = DataIngestion.initiate_data_ingestion()

---

## data_transformation.py
- Imports:
  - os, sys, numpy, pandas, train_test_split, dataclass,
  - ColumnTranser, SimpleImputer, Pipeline, OneHotEncoder, StandardScaler
  - CustomException, logging, save_object
  - DataTransformationConfig, DataTransformation,
- DataTransformationConfig:
  - preprocessor_obj_file_path: artifacts/preprocessor.pkl
- DataTransformation:
  - __init__(self):
    - data_transformation_config = preprocessor_obj_file_path
  - get_data_transformer_object(self)
    - try:
      - list numerical columns (excluding target feature)
      - list categorical columns (excluding target feature)
      - set steps on num_pipeline: impute (with median) and scale (with StandardScaler)
      - set steps on cat_ pipeline: impute (with most-frequent), encode (with OHE), scale (with StandardScaler[with_mean=False])
      - combines different preprocessing pipelines for different column types as preprocessor
      - return preprocessor
    - except
      - raise CustomException
  - initiate_data_transformation(self, train_path, test_path):
    - try
      - create train_df from train.csv
      - create test_df from test.csv
      - obtain preprocessing_obj = self.get_data_transformer_object()
      - set target_column name
      - seperate input and target features from train and test data frame
        - input_features_train_df = train_df.drop(columns=[target_column_name])
        - target_feature_train_df = train_df[target_column_name]
        - input_features_test_df = test_df.drop(columns=[target_column_name])
        - target_feature_test_df = test_df[target_column_name]
      - Fits the preprocessing pipeline on training data and transforms it into model-ready numerical features.
      - Applies the same preprocessing rules learned from training data to the test data.
      - Merges processed training features and target values into one array by adding the target as the last column.
      - Combines test features and test labels into one array by adding the target as the last column.
      - Save the trained preprocessing pipeline so the same transformations can be reused during inference.
      - Returns processed training & test data along with the path to the saved preprocessing object.

__main__:
	train_data, test_data = DataIngestion.initiate_data_ingestion()


---

## model_trainer.py
- Imports:
  - os, sys, numpy, pandas, train_test_split, dataclass,
  - CatBoostRegressor, CatBoostRegressorWrapper
  - AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor,
  - LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, XGBRegressor
  - CustomException, logging, 
  - save_object, evaluate_models
  - DataTransformationConfig, DataTransformation,
- ModelTrainerConfig:
  - trained_model_file_path: artifacts/model.pkl
- ModelTrainer:
  -  __init__(self):
    - model_trainer_config = trained_model_file_path
  - initiate_model_trainer(self, train_array, test_array):
    - try:
      - Split Feature–target from already prepared NumPy arrays.
      - Create a dictionary of models
      - Create a dictionary of hyper-parameters
      - Evaluate multiple ML models on training and testing data using given hyperparameters, and returns a dictionary summarizing each model’s performance.
      - Pick the highest performance score among all the models.
      - Find the model name in model_report that has the highest performance score.
      - Pick out the best-performing model object from your models dictionary so you can use it in your pipeline.
      - Prevent using a poorly performing model by throwing an error if the best model is below a threshold.
      - Store your best-performing model to a file so it can be loaded and used later in production or for inference.
      - Predict: Compute outputs from your model for test data.
      - Evaluate: Measure how accurate those predictions are using R².
      - Print the best model name and best model score
      - Return r2 score
    - except:

---

## utils.py
- Imports:
  - os, sys, numpy, pandas, dill, 
  - 2r_score, GridSearchCV
  - CustomException
  - BaseEstimator, RegressorMixin, catboost

- CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
  - __init__(self, iterations=1000, learning_rate=0.03, depth=6, 
                 l2_leaf_reg=3.0, random_strength=1.0, 
                 verbose=False, **kwargs):
  - fit(self, X, y, **fit_params):
  - predict(self, X):
  - get_params(self, deep=True):
  - set_params(self, **params):

- save_object(file_path, obj):
  - try:
    - Set dir_path from the file_path
    - Make directory with dir_path
    - Store the Python object into a binary file so you can load it later.
  - except:
  
- evaluate_models(X_train,y_train,X_test,y_test,models,params):
  - try:
    - Create a empty dictionary report
    - Run a loop to the length of models
      - Get model from list of models
      - Get paras from model keys??
      - Create a GridSearch object for hyper-parameter tuning.
      - Run grid search: train many models with different hyper-parameters using cross-validation and store the best trained model inside gs.
      - Take the best hyper-parameters from GridSearch and apply them to your original model object
      - Learn from the training data so the model can predict targets from features.
      - Predicts targets for the training set
      - Predicts targets for the test set
      - Evaluates your model on the training data, it tells you how well your model fits the data it has already seen
      - Evaluates your model on unseen test data, that measures generalization
      - Store the test score of the i-th model in the report dictionary, using the model’s name as the key.
      - Return the report
  - except:

- load_object(file_path):
  - try:
    - Read a binary file from disk and restore the Python object inside it.
    - Open a saved binary file and bring back the Python object into memory exactly as it was before saving.
  - except:

# Pipeline

## train_pipeline.py
- Imports
  - sys
  - DataIngestion, DataTransformation, ModelTrainer
  - CustomException, logging
  
- TrainPipeline:
  - __init__(self): 
  - initiate_training(self):
    - train_path, test_path = self.data_ingestion.initiate_data_ingestion()
    - train_arr, test_arr, preprocessor_path = \
                self.data_transformation.initiate_data_transformation(
                    train_path=train_path,
                    test_path=test_path
                )
    - r2_score = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
- if __name__ == "__main__":
  - results = TrainPipeline().initiate_training()

## train.py
- Imports
  - os, sys
  - logging, TrainPipeline

- main():
  - try:
    - set raw_data_path
    - if raw_data_path doesn't exist:
      - raise FileNotFoundError
    - make artifacts directory
    - results = TrainPipeline().initiate_training()
    - Show result metric
      - Print R2 Score
      - Print Model Path
      - Print Preprocessor Path
      - Print Train data Path
      - Print Test data Path
---

## predict_pipepeline.py
- Imports
  - sys, pandas
  - CustomException, load_object

- PredictPipeline:
  - __init__(self):

  - predict(self, features):
    - try:
      - Set model_path
      - Set preprocessor_path
      - Load model
      - Load preprocessor
      - Apply the trained preprocessing pipeline to input features and returns the transformed (scaled/encoded) data ready for the ML model.
      - Generate the model’s predicted target values for the given (already preprocessed) input data.
      - Return the predicted target values

- CustomData:
  - __init__(self, 
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int
                ):
  - get_data_as_data_frame(self):
    - Create custom_data_input dictonary
    - Return dataframe from custom data input dict