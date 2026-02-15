"""
Hyperparameter Overrides Configuration

This file allows you to customize hyperparameter grids for model tuning.
Only models defined here will override the defaults in ModelTuner.
Models not listed here will use the default parameter grids.

Usage:
    - To customize a model's hyperparameters, add it to CLF_PARAM_GRIDS or REG_PARAM_GRIDS
    - To use all defaults, leave the dictionaries empty: CLF_PARAM_GRIDS = {}
    - Partial overrides are supported: only specify the models you want to customize

Example:
    # Override only RandomForest for regression, use defaults for all others
    REG_PARAM_GRIDS = {
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
        }
    }
"""

# =============================================================================
# CLASSIFICATION HYPERPARAMETER OVERRIDES
# =============================================================================
# Uncomment and modify to override default classification hyperparameters
# Leave empty {} to use all defaults from ModelTuner

CLF_PARAM_GRIDS = {
    # 'RandomForest': {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    # },
    # 'DecisionTree': {
    #     'max_depth': [None, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'criterion': ['gini', 'entropy'],
    # },
    # 'SVC': {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.01, 0.1],
    #     'kernel': ['rbf', 'poly', 'sigmoid'],
    # },
    # 'KNN': {
    #     'n_neighbors': [3, 5, 7, 9, 11],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan'],
    # },
    # 'LogisticRegression': {
    #     'C': [0.01, 0.1, 1, 10],
    #     'solver': ['lbfgs', 'liblinear'],
    #     'max_iter': [500, 1000],
    # },
    # 'NaiveBayes': {
    #     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
    # },
    # 'Bagging': {
    #     'n_estimators': [10, 25, 50],
    #     'max_samples': [0.5, 0.7, 1.0],
    #     'max_features': [0.5, 0.7, 1.0],
    # },
    # 'Stacking': [
    #     {'final_estimator__C': [0.1, 1, 10]},
    # ],
}


# =============================================================================
# N_ITER OVERRIDES FOR RANDOMIZED SEARCH (per model)
# =============================================================================
# Override the default n_iter values for RandomizedSearchCV
# Only applies when SEARCH_TYPE = 'random' in config.py
# Leave empty {} to use defaults from ModelTuner

CLF_N_ITER = {
    # 'RandomForest': 30,        # Grid: ~900 combinations
    # 'DecisionTree': 25,        # Grid: ~360 combinations
    # 'SVC': 20,                 # Grid: ~36 combinations
    # 'KNN': 20,                 # Grid: ~36 combinations
    # 'LogisticRegression': 25,  # Grid: ~52 combinations (list of dicts)
    # 'NaiveBayes': 8,           # Grid: 8 combinations (exhaustive)
    # 'Bagging': 15,             # Grid: ~27 combinations
    # 'Stacking': 2,             # Grid: 2 combinations (exhaustive)
}

REG_N_ITER = {
    # 'RandomForest': 30,        # Grid: ~900 combinations
    # 'DecisionTree': 25,        # Grid: ~360 combinations
    # 'SVM': 20,                 # Grid: ~96 combinations
    # 'KNN': 20,                 # Grid: ~36 combinations
    # 'LinearRegression': 2,     # Grid: 2 combinations (exhaustive)
    # 'Ridge': 15,               # Grid: ~16 combinations
    # 'Lasso': 10,               # Grid: ~8 combinations
    # 'ElasticNet': 15,          # Grid: ~32 combinations
    # 'GradientBoosting': 30,    # Grid: ~300 combinations
    # 'XGBoost': 25,             # Grid: ~108 combinations
    # 'LightGBM': 25,            # Grid: ~108 combinations
    # 'Bagging': 15,             # Grid: ~36 combinations
    # 'Stacking': 2,             # Grid: 2 combinations (exhaustive)
}


# =============================================================================
# REGRESSION HYPERPARAMETER OVERRIDES
# =============================================================================
# Uncomment and modify to override default regression hyperparameters
# Leave empty {} to use all defaults from ModelTuner

REG_PARAM_GRIDS = {
    # 'RandomForest': {
    #     'n_estimators': [10, 20, 100],
    #     'max_depth': [3, 5, 7],
    #     'min_samples_split': [8, 16, 32],
    #     'min_samples_leaf': [1, 2, 4],
    # },
    # 'DecisionTree': {
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2, 4],
    #     'criterion': ['squared_error', 'friedman_mse'],
    # },
    # 'SVM': {
    #     'C': [1, 10, 100],
    #     'gamma': ['scale', 'auto'],
    #     'kernel': ['rbf'],
    #     'epsilon': [0.01, 0.1],
    # },
    # 'KNN': {
    #     'n_neighbors': [3, 5, 7, 9],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan'],
    # },
    # 'Ridge': {
    #     'alpha': [0.01, 0.1, 1, 10, 100],
    # },
    # 'Lasso': {
    #     'alpha': [0.001, 0.01, 0.1, 1],
    # },
    # 'ElasticNet': {
    #     'alpha': [1, 10, 100],
    #     'l1_ratio': [0.2, 0.5, 0.8, 1.0],
    # },
    # 'GradientBoosting': {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    # },
    # 'XGBoost': {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    # },
    # 'LightGBM': {
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'max_depth': [3, 5, 7],
    # },
    # 'Bagging': {
    #     'n_estimators': [10, 25, 50],
    #     'max_samples': [0.5, 0.7, 1.0],
    #     'max_features': [0.5, 0.7, 1.0],
    # },
    # 'Stacking': [
    #     {'final_estimator__alpha': [0.1, 1, 10]},
    # ],
}
