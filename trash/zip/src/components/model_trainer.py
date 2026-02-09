import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from catboost import CatBoostRegressor
from src.utils import CatBoostRegressorWrapper  # Import the wrapper
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    # def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data.")

            # Split Feature–target from already prepared NumPy arrays.
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],

                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create a dictionary of models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),

                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "CatBoosting Regressor": CatBoostRegressorWrapper(verbose=False),  # Use wrapper instead

                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Create a dictionary of hyper-parameters
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],

                    # 'depth': [6, 8],           # ✓ Must match wrapper's __init__ parameter name
                    # 'learning_rate': [0.01, 0.1],
                    # 'iterations': [100, 200],
                    # 'l2_leaf_reg': [1.0, 3.0]  # Optional but valid
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models)

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, models=models, params = params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Pick out the best-performing model object from your models dictionary so you can use it in your pipeline.
            best_model = models[best_model_name]

            # Prevent using a poorly performing model by throwing an error if the best model is below a threshold.
            if best_model_score < 0.6:
                raise CustomException("No best model found.", sys)
            
            logging.info(f"Best found model on both training and testing dataset.")

            # stores your best-performing model to a file so it can be loaded and used later in production or for inference.
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # Predict: Compute outputs from your model for test data.
            # Evaluate: Measure how accurate those predictions are using R².
            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)

            print(f"Best Model: {best_model_name}, Score: {best_model_score}")
            
            return r2score

        except Exception as e:
            raise CustomException (e, sys)