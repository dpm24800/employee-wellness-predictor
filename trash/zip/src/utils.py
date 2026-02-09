import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
# from src.components import ModelTrainer #

# ===== ADD THIS AT THE TOP OF utils.py (after imports) =====
from sklearn.base import BaseEstimator, RegressorMixin
import catboost

class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Fully sklearn-compatible wrapper for CatBoostRegressor (supports GridSearchCV)
    """
    def __init__(self, iterations=1000, learning_rate=0.03, depth=6, 
                 l2_leaf_reg=3.0, random_strength=1.0, 
                 verbose=False, **kwargs):
        # Store ALL tunable parameters as direct attributes (required for sklearn)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.verbose = verbose
        self.kwargs = kwargs  # Extra parameters not in signature
        
        self.model = None  # Will be initialized in fit()
    
    def fit(self, X, y, **fit_params):
        # Rebuild model with current parameters (critical for GridSearchCV)
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_strength': self.random_strength,
            'verbose': self.verbose,
            **self.kwargs
        }
        self.model = catboost.CatBoostRegressor(**params)
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        # Return ALL parameters (required for GridSearchCV)
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'random_strength': self.random_strength,
            'verbose': self.verbose,
        }
        if deep:
            params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        # Update parameters dynamically (required for GridSearchCV)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# def evaluate_models(X_train, y_train, X_test, y_test, models):
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            '''
            Take the best hyper-parameters from GridSearch
            Apply them to your original model object
            Now model is ready to train (or refit) with the best hyper-parameters.
            '''
            model.set_params(**gs.best_params_)  # Manual refit (GOOD)
            
            # Learn from the training data so the model can predict targets from features.
            model.fit(X_train, y_train) # Train model
            
            # Predicts targets for the training set
            # Predicts targets for the test set
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluates your model on the training data, it tells you how well your model fits the data it has already seen
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score of the i-th model in the report dictionary, using the model’s name as the key.
            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException (e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)