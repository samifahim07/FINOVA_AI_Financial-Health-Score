"""scikit-learn-compatible wrapper for CatBoost on recent sklearn versions.

CatBoost 1.2.x may not expose the estimator tags used by sklearn >= 1.6.
This wrapper ensures the fitted pipeline can predict and be reloaded.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from catboost import CatBoostRegressor


class CompatibleCatBoostRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=350, depth=5, learning_rate=0.055,
                 random_seed=42, thread_count=4):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function='RMSE',
            verbose=False,
            allow_writing_files=False,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
        )
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'model_')
        return self.model_.feature_importances_
