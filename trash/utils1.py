# ===== ADD THIS WRAPPER AT THE TOP OF utils.py =====
from sklearn.base import BaseEstimator, RegressorMixin
import catboost

class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper to make CatBoostRegressor compatible with scikit-learn >=1.6
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = catboost.CatBoostRegressor(**kwargs)
        self.estimator_type = "regressor"  # Critical for sklearn compatibility
    
    def fit(self, X, y, **fit_params):
        # Handle CatBoost-specific fit parameters (like eval_set)
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    # Delegate other attributes/methods to inner model
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.model, name)
# =====================================================
