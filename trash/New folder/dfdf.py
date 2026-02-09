import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# -----------------------
# Feature / target split
# -----------------------
df = pd.read_csv('work_from_home_burnout_dataset.csv')

X = df.drop(columns=["user_id", "burnout_score", "burnout_risk"])

y_reg = df["burnout_score"]
y_clf = df["burnout_risk"]

cat_cols = ["day_type"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# -----------------------
# Regression pipeline
# -----------------------

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

reg_pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

reg_pipe.fit(Xr_train, yr_train)

yr_pred = reg_pipe.predict(Xr_test)

print("Regression R2 :", r2_score(yr_test, yr_pred))
# print("RMSE :", mean_squared_error(yr_test, yr_pred, squared=False))


# -----------------------
# Classification pipeline
# -----------------------

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_pipe = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

clf_pipe.fit(Xc_train, yc_train)

yc_pred = clf_pipe.predict(Xc_test)

print(classification_report(yc_test, yc_pred))
