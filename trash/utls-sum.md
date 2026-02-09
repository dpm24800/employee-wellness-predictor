from sklearn.model_selection import GridSearchCV
r2_score=  x
pickle = y


```py

# src/utils.py
def save_object(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    report = {}
    for name, model in models.items():
        gs = GridSearchCV(model, params[name], cv=3)
        gs.fit(X_train, y_train)
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        report[name] = r2_score(y_test, y_test_pred)
    return report

def load_object(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
```