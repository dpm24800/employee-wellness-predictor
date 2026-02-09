# ============================================================
# Burnout Score – EDA + Multiple Regression Models
# File  : clean.csv
# Target: burnout_score
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv("clean.csv")

# ------------------------------------------------------------
# ------------------------ EDA -------------------------------
# ------------------------------------------------------------

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isna().sum())

print("\nStatistical summary:")
print(df.describe())

# Categorical distribution
print("\nDay type distribution:")
print(df["day_type"].value_counts())

# Correlation (numerical only)
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

# Target distribution
plt.figure(figsize=(6, 4))
sns.histplot(df["burnout_score"], kde=True)
plt.title("Burnout score distribution")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# --------------------- Train / Test -------------------------
# ------------------------------------------------------------

TARGET = "burnout_score"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# ------------------ Preprocessing ---------------------------
# ------------------------------------------------------------

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------
# -------------------- Models --------------------------------
# ------------------------------------------------------------

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "ElasticNet": ElasticNet(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
}

# ------------------------------------------------------------
# ------------------ Training & Evaluation -------------------
# ------------------------------------------------------------

results = []

for name, model in models.items():

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "model": name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    })

    print(f"\n{name}")
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

# ------------------------------------------------------------
# ------------------ Result table ----------------------------
# ------------------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    by="r2", ascending=False
).reset_index(drop=True)

print("\n================ Model Comparison ================\n")
print(results_df)


# ------------------------------------------------------------
# -------- Optional: Cross-validated R2 of best model --------
# ------------------------------------------------------------

best_model_name = results_df.loc[0, "model"]
best_model = models[best_model_name]

best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])

cv_scores = cross_val_score(
    best_pipeline,
    X,
    y,
    scoring="r2",
    cv=5
)

print(f"\nBest model: {best_model_name}")
print("CV R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())
