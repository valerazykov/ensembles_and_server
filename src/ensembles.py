import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


class RandomForestMSE:
    def __init__(
        self, n_estimators=100, max_depth=None,
        feature_subsample_size=None,
        error_func=root_mean_squared_error,
        random_state=42,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third
            of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.error_func = error_func
        self.random_state = random_state
        self.trees_parameters = trees_parameters

        self.trees = [
            DecisionTreeRegressor(max_depth=max_depth, **trees_parameters)
            for _ in range(n_estimators)
        ]

        np.random.seed(random_state)

        self.obj_idx = None
        self.feat_idx = None
        self.ensemble_errors_history = None

    def _get_obj_idx(self, X):
        idx = np.zeros((self.n_estimators, X.shape[0]), dtype="int64")
        for i in range(self.n_estimators):
            idx[i] = np.random.choice(X.shape[0], X.shape[0], replace=True)
        return idx

    def _get_feat_idx(self, X):
        if self.feature_subsample_size is None:
            k = max(1, X.shape[1] // 3)
        else:
            k = max(1, X.shape[1] * self.feature_subsample_size)

        idx = np.zeros((self.n_estimators, k), dtype="int64")
        for i in range(self.n_estimators):
            idx[i] = np.random.choice(X.shape[1], k, replace=False)
        return idx

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """

        self.obj_idx = self._get_obj_idx(X)
        self.feat_idx = self._get_feat_idx(X)

        for tree_num, tree in enumerate(self.trees):
            X_train = X[self.obj_idx[tree_num], :][:, self.feat_idx[tree_num]]
            tree.fit(X_train, y[self.obj_idx[tree_num]])

        if X_val is not None and y_val is not None:
            trees_preds = np.zeros((self.n_estimators, X_val.shape[0]))
            for tree_num, tree in enumerate(self.trees):
                trees_preds[tree_num] = tree.predict(
                    X_val[:, self.feat_idx[tree_num]]
                )

            ensemble_preds = (np.cumsum(trees_preds, axis=0) /
                              np.arange(
                                  1, self.n_estimators + 1
                              ).reshape(-1, 1))

            self.ensemble_errors_history = np.zeros(self.n_estimators)
            for i, ens_pred in enumerate(ensemble_preds):
                self.ensemble_errors_history[i] = self.error_func(
                    y_val, ens_pred
                )

        return self

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """

        trees_preds = np.zeros((self.n_estimators, X.shape[0]))
        for tree_num, tree in enumerate(self.trees):
            trees_preds[tree_num] = tree.predict(
                X[:, self.feat_idx[tree_num]]
            )

        ens_pred = np.mean(trees_preds, axis=0)
        return ens_pred


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5,
        feature_subsample_size=None, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third
            of all features.
        """
        pass

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pass
