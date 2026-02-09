
## Package Installation
```py
pip install imbalanced-learn
```

## Package Imports
```py
# Imports
import numpy as np
import pandas as pd


# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score


# Scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler  # for scaling 0-1

# Encoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# Classification
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB




# Model Evaluation
# Regression 
# Classification algorithms: ACC-FPR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score,  
#

import warnings
warnings.filterwarnings('ignore')
```






```py
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify = y
)


print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:", cm = confusion_matrix(y_true, y_pred)
print("Classification Report:", classification_report(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))   # IMPORTANT FOR CHURN



# Define Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=500, solver="liblinear"),
    "SVC": SVC(kernel="rbf", random_state=42),
    "GaussianNB": GaussianNB(),
}

# Cross validation: in order
cv  = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)


# -------------------------------------
# training with smote 
for name, model in models.items():
    print(f"\nRunning model: {name}")
    
    pipe = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Cross-validation (Recall is better for churn)
    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring='recall',
        n_jobs=1
    )
    
    print(f"CV Recall (Churn): Mean={cv_scores.mean():.4f}")
    
    # Train full model
    pipe.fit(X_train, y_train)
    best_model[name] = pipe

  
    # Test evaluation
    y_test_pred = pipe.predict(X_test)
    test_recall = recall_score(y_test, y_test_pred)
    test_scores[name] = test_recall

    # ---- Training accuracy ----
    y_train_pred = pipe.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")
    
    
    print("Test Set Performance:")
    modelperformance(y_test, y_test_pred, model_name=name)
# -------------------------------------
# Loop over models

for name, model in models.items():
    print(f"\nRunning model: {name}")
    
    # Pipeline (no scaler needed for GaussianNB)
    pipe = Pipeline(steps=[
        ('classifier', model)
    ])
    
    # Cross-validation
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    results[name] = scores
    print(f"CV Accuracy: Mean={scores.mean():.4f} | Std={scores.std():.4f}")
    
    # Train on full training data
    pipe.fit(X_train, y_train)
    best_models[name] = pipe
    
   
    # Training Accuracy
    y_train_pred = pipe.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("Training Accuracy:", train_acc)
    

    # Test Evaluation
    y_test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_accuracies[name] = test_acc
    print("Test Accuracy:", test_acc)
    
    # Confusion matrix & metrics
    modelperformance(y_test, y_test_pred, model_name=name)

# -------------------------------------


# Print best model

best_model_name = max(test_scores, key=test_scores.get)
best_model = best_model[best_model_name]

print("\n==============================")
print(f"Best Model (Based on Churn Recall): {best_model_name}")
print(f"Best Recall: {test_scores[best_model_name]:.4f}")
print("==============================")



best_model_name = max(test_accuracies, key=test_accuracies.get)
best_model_acc = test_accuracies[best_model_name]

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Test Accuracy: {best_model_acc:.4f}")
print("==============================")

# --------------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

data = load_iris()
X_train, X_test, y_train,y_test = train_test_split(data.data, data.target, test_size = 0.2, random_state = 42)

param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 10.0]}
naive_bayes_classifier = MultinomialNB()
clf = GridSearchCV(naive_bayes_classifier, param_grid, cv = 5)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(" Best hyperparameters:(clf.best_params.)")
# --------------------------------------------------------------------------------
```