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

        error_func: callable
            Function for measuring errors when the number of trees increases.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.error_func = error_func
        self.random_state = random_state
        self.trees_parameters = trees_parameters

        self.trees = None
        self.ensemble_errors_history = None

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

        np.random.seed(self.random_state)

        max_features = (self.feature_subsample_size
                        if self.feature_subsample_size is not None
                        else max(1, X.shape[1] // 3))

        self.trees = [
            DecisionTreeRegressor(criterion="squared_error",
                                  splitter="random",
                                  max_depth=self.max_depth,
                                  max_features=max_features,
                                  random_state=self.random_state,
                                  **self.trees_parameters)
            for _ in range(self.n_estimators)
        ]

        for tree in self.trees:
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree.fit(X[idx], y[idx])

        if X_val is not None and y_val is not None:
            self.ensemble_errors_history = np.zeros(self.n_estimators)
            trees_preds = np.zeros((self.n_estimators, X_val.shape[0]))
            for tree_num, tree in enumerate(self.trees):
                trees_preds[tree_num] = tree.predict(X_val)
                ensemble_pred = np.sum(trees_preds, axis=0) / (tree_num + 1)
                self.ensemble_errors_history[tree_num] = self.error_func(
                    y_val, ensemble_pred
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
            trees_preds[tree_num] = tree.predict(X)

        return np.mean(trees_preds, axis=0)


class GradientBoostingMSE:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=5,
        feature_subsample_size=None,
        error_func=root_mean_squared_error,
        random_state=42,
        **trees_parameters
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

        error_func: callable
            Function for measuring errors when the number of trees increases.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.error_func = error_func
        self.random_state = random_state
        self.trees_parameters = trees_parameters

        self.trees = None
        self.alphas = None
        self.ensemble_errors_history = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        max_features = (self.feature_subsample_size
                        if self.feature_subsample_size is not None
                        else max(1, X.shape[1] // 3))

        self.trees = [
            DecisionTreeRegressor(criterion="squared_error",
                                  splitter="random",
                                  max_depth=self.max_depth,
                                  max_features=max_features,
                                  random_state=self.random_state,
                                  **self.trees_parameters)
            for _ in range(self.n_estimators)
        ]

        self.alphas = np.zeros(self.n_estimators)
        ensemble_pred_prev = np.zeros(X.shape[0])
        tree_pred = np.zeros_like(ensemble_pred_prev)

        def func(alpha):
            return np.sum((ensemble_pred_prev + alpha * tree_pred - y) ** 2)

        for tree_num, tree in enumerate(self.trees):
            tree.fit(X, y - ensemble_pred_prev)
            tree_pred = tree.predict(X)
            alpha = minimize_scalar(func).x
            self.alphas[tree_num] = self.learning_rate * alpha
            ensemble_pred_prev += self.learning_rate * alpha * tree_pred

        if X_val is not None and y_val is not None:
            self.ensemble_errors_history = np.zeros(self.n_estimators)
            trees_preds_val = np.zeros((self.n_estimators, X_val.shape[0]))
            for tree_num, tree in enumerate(self.trees):
                trees_preds_val[tree_num] = tree.predict(X_val)
                ensemble_pred = np.sum(
                    trees_preds_val * self.alphas.reshape(-1, 1), axis=0
                )
                self.ensemble_errors_history[tree_num] = self.error_func(
                    y_val, ensemble_pred
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
            trees_preds[tree_num] = tree.predict(X)

        return np.sum(trees_preds * self.alphas.reshape(-1, 1), axis=0)
