"""
Hyperparameter Tuning Module

This module handles hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Try to import hyperparameter overrides (optional)
try:
    from hyperparameters import CLF_PARAM_GRIDS as CLF_OVERRIDES, REG_PARAM_GRIDS as REG_OVERRIDES
    _HAS_OVERRIDES = True
except ImportError:
    CLF_OVERRIDES = {}
    REG_OVERRIDES = {}
    _HAS_OVERRIDES = False

# Try to import n_iter overrides (optional)
try:
    from hyperparameters import CLF_N_ITER as CLF_N_ITER_OVERRIDES, REG_N_ITER as REG_N_ITER_OVERRIDES
except ImportError:
    CLF_N_ITER_OVERRIDES = {}
    REG_N_ITER_OVERRIDES = {}
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor,
                              BaggingClassifier, BaggingRegressor, StackingClassifier, StackingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb


class ModelTuner:
    """
    Handles hyperparameter tuning for ML models

    This class provides methods to tune different models using GridSearchCV
    or RandomizedSearchCV with predefined parameter grids.

    Parameter Grid Priority:
        1. Custom param_grid passed to tune_model() (highest priority)
        2. Override grids defined in hyperparameters.py
        3. Default grids defined in this class (CLF_PARAM_GRIDS, REG_PARAM_GRIDS)

    To customize hyperparameters without modifying this file, edit hyperparameters.py
    """

    # Per-model n_iter for RandomizedSearchCV
    # Each value is tuned relative to the model's parameter grid size
    # Models with small grids get fewer iterations (no point exceeding total combinations)
    # Models with large grids get more iterations for better coverage
    CLF_N_ITER = {
        'RandomForest': 30,        # Grid: ~900 combinations
        'DecisionTree': 25,        # Grid: ~360 combinations
        'SVC': 20,                 # Grid: ~36 combinations
        'KNN': 20,                 # Grid: ~36 combinations
        'LogisticRegression': 25,  # Grid: ~52 combinations (list of dicts)
        'NaiveBayes': 8,           # Grid: 8 combinations (exhaustive)
        'Bagging': 15,             # Grid: ~27 combinations
        'Stacking': 2,             # Grid: 2 combinations (exhaustive)
    }

    REG_N_ITER = {
        'RandomForest': 30,        # Grid: ~900 combinations
        'DecisionTree': 25,        # Grid: ~360 combinations
        'SVM': 20,                 # Grid: ~96 combinations
        'KNN': 20,                 # Grid: ~36 combinations
        'LinearRegression': 2,     # Grid: 2 combinations (exhaustive)
        'Ridge': 15,               # Grid: ~16 combinations
        'Lasso': 10,               # Grid: ~8 combinations
        'ElasticNet': 15,          # Grid: ~32 combinations
        'GradientBoosting': 30,    # Grid: ~300 combinations
        'XGBoost': 25,             # Grid: ~108 combinations
        'LightGBM': 25,            # Grid: ~108 combinations
        'Bagging': 15,             # Grid: ~36 combinations
        'Stacking': 2,             # Grid: 2 combinations (exhaustive)
    }

    # Classification parameter grids
    # Depending on the model and its tuned hyperparameters, these grids can be adjusted to improve performance and speed up tuning
    # We can reduce the number of options for faster tuning or expand them for better results depending on your machine capabilities
    CLF_PARAM_GRIDS = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 6, 7, 10],
            'min_samples_split': [2, 5, 10, 12],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'max_features': ['sqrt', 'log2', None]
        },
        'DecisionTree': {
            'max_depth': [None, 5, 6, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        },
        'SVC': {
            'C': [1, 10, 100],                    
            'gamma': ['scale', 0.01, 0.1],        
            'kernel': ['rbf', 'linear'],          
            'class_weight': [None, 'balanced']   
        },
        #'SVC': {
        #    'C': [100],                    
        #    'gamma': ['scale'],        
        #    'kernel': ['rbf'],          
        #    'class_weight': ['balanced']   
        #},
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'LogisticRegression': [
            # saga and liblinear solvers support l1, l2, elasticnet, and none penalties with different constraints:
            # Note: Increased max_iter to 5000 to avoid ConvergenceWarning
            # Note: When penalty=None, C is ignored (no regularization), so we don't include C in those grids

            # lbfgs: only l2 or None (fast, good for small datasets)
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs'],
             'max_iter': [5000], 'class_weight': [None, 'balanced']},
            {'penalty': [None], 'solver': ['lbfgs'],  # No C when penalty=None
             'max_iter': [5000], 'class_weight': [None, 'balanced']},

            # liblinear: l1 or l2 (no elasticnet, no None) - good for high-dimensional sparse data
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear'],
             'max_iter': [5000], 'class_weight': [None, 'balanced']},
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear'],
             'max_iter': [5000], 'class_weight': [None, 'balanced']},

            # saga: all penalties (l1, l2, elasticnet, None) - best for large datasets
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['saga'],
             'max_iter': [5000], 'class_weight': [None, 'balanced']},
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['saga'],
             'max_iter': [5000], 'class_weight': [None, 'balanced']},
            {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'],
             'l1_ratio': [0.3, 0.5, 0.7], 'max_iter': [5000], 'class_weight': [None, 'balanced']},
            {'penalty': [None], 'solver': ['saga'],  # No C when penalty=None
             'max_iter': [5000], 'class_weight': [None, 'balanced']},
        ],
        'NaiveBayes': {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],  # Extended range (more values)
            'priors': [None]  # Could add custom priors based on class distribution
        },
        'Bagging': {
            'n_estimators': [25, 50, 100],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.6, 0.8, 1.0]
        },
        'Stacking': {
            # Stacking hyperparameters are minimal (cv folds mainly)
            # Base estimators use default parameters
            'cv': [3, 5]
        }
    }

    # Regression parameter grids
    REG_PARAM_GRIDS = {
        'RandomForest': {
            'n_estimators': [30, 50, 100, 200, 250],
            'max_depth': [None, 5, 6, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6, 8],
        },
        'DecisionTree': {
            'max_depth': [None, 5, 6, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
            'max_features': ['sqrt', 'log2', None]
        },
        # 'DecisionTree': {
        #     'max_depth': [None, 5, 6],
        #     'min_samples_split': [2, 5],
        #     'min_samples_leaf': [1, 2, 4],
        #     'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        #     'max_features': ['sqrt', 'log2', None]
        # },
        'SVM': {
            'C': [0.1, 1, 10, 100],  # Range for regularization (higher C = less regularization)
            'gamma': ['scale', 'auto', 0.01, 0.1],  # Kernel coefficient
            'kernel': ['rbf', 'linear'],  # RBF and linear kernels
            'epsilon': [0.1, 0.2, 0.5],  # Epsilon-tube width (wider = more tolerance)
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'LinearRegression': {
            # LinearRegression has minimal hyperparameters
            # Using fit_intercept as a placeholder to enable grid search
            'fit_intercept': [True],  # Whether to calculate intercept (almost always True)
            'copy_X': [True],  # Keep default to preserve input data
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'solver': ['svd', 'cholesky', 'lsqr', 'saga'],
            'max_iter': [10000]  # Increased to avoid ConvergenceWarning (for saga solver only)
        },
        'Lasso': {
            # Note: Very small alpha (weak regularization) can cause convergence issues
            # Increased min alpha from 1e-7 to 0.001 for better convergence
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'selection': ['cyclic', 'random'],
            'max_iter': [10000],
            'tol': [1e-4]  # Default tolerance
        },
        'ElasticNet': {
            # Note: Very small alpha (weak regularization) can cause convergence issues
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.7, 0.9],
            'selection': ['cyclic', 'random'],
            'max_iter': [10000],
            'tol': [1e-4]  # Default tolerance
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200, 250],
            'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200, 250],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5]
        },
        'LightGBM': {
            'n_estimators': [50, 100, 200, 250],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100]
        },
        'Bagging': {
            'n_estimators': [25, 50, 100, 200],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.6, 0.8, 1.0]
        },
        'Stacking': {
            # Stacking hyperparameters are minimal (cv folds mainly)
            # Base estimators use default parameters
            'cv': [3, 5]
        }
    }

    def __init__(self, random_state=42):
        """
        Initialize the tuner

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def tune_model(self, X_train, y_train, model_type='classification',
                   model_name='RandomForest', search_type='grid',
                   cv=5, param_grid=None, verbose=True):
        """
        Tune hyperparameters for a specified model

        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'classification' or 'regression'
            model_name: Model name ('RandomForest', 'DecisionTree', 'SVC'/'SVM', 'KNN')
            search_type: 'grid' or 'random' search
            cv: Number of cross-validation folds
            param_grid: Custom parameter grid (optional, uses default if None)
            verbose: Whether to print detailed output

        Returns:
            tuple: (best_model, best_params, best_score)

        Example:
            >>> tuner = ModelTuner()
            >>> best_model, params, score = tuner.tune_model(
            ...     X_train, y_train,
            ...     model_type='classification',
            ...     model_name='RandomForest'
            ... )
        """
        if verbose:
            print("="*80)
            print(f" "*15 + f"TUNING {model_name.upper()} ({model_type.upper()})")
            print("="*80)

        # Get base model and param grid
        base_model, param_grid = self._get_model_and_params(
            model_type, model_name, param_grid
        )

        # Choose scoring metric
        # Use f1_macro for classification to optimize for balanced performance across all classes
        # This is better than accuracy for multi-class problems, especially with class imbalance
        scoring = 'f1_macro' if model_type == 'classification' else 'r2'

        if verbose:
            total_combinations = self._count_combinations(param_grid)
            print(f"\nConfiguration:")
            print(f"  Search type: {search_type}")
            print(f"  CV folds: {cv}")
            print(f"  Scoring: {scoring}")
            param_source = "hyperparameters.py (override)" if getattr(self, '_last_override_used', False) else "default"
            print(f"  Parameter grid: {param_source}")
            print(f"  Parameter grid combinations: {total_combinations}")
            if search_type == 'random':
                grid_name = 'SVM' if (model_type == 'regression' and model_name == 'SVC') else model_name
                if model_type == 'classification':
                    n_iter = CLF_N_ITER_OVERRIDES.get(grid_name, self.CLF_N_ITER.get(grid_name, 20))
                else:
                    n_iter = REG_N_ITER_OVERRIDES.get(grid_name, self.REG_N_ITER.get(grid_name, 20))
                n_iter = min(n_iter, total_combinations)
                print(f"  Random search n_iter: {n_iter} / {total_combinations}")

        # Create search object
        if search_type == 'grid':
            search_cv = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0,
                error_score='raise'
            )
        else:  # random search
            # Get per-model n_iter: check overrides first, then defaults
            grid_name = 'SVM' if (model_type == 'regression' and model_name == 'SVC') else model_name
            if model_type == 'classification':
                n_iter = CLF_N_ITER_OVERRIDES.get(grid_name, self.CLF_N_ITER.get(grid_name, 20))
            else:
                n_iter = REG_N_ITER_OVERRIDES.get(grid_name, self.REG_N_ITER.get(grid_name, 20))

            # Cap n_iter to total combinations (no point sampling more than exists)
            total_combinations = self._count_combinations(param_grid)
            n_iter = min(n_iter, total_combinations)

            search_cv = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state,
                error_score='raise'
            )

        if verbose:
            print(f"\nData shape:")
            print(f"  X_train: {X_train.shape}")
            print(f"  y_train: {y_train.shape}")
            print(f"\nStarting {search_type} search...")

        # Fit the search
        try:
            search_cv.fit(X_train, y_train)
        except Exception as e:
            print(f"\n❌ ERROR during {search_type} search: {e}")
            print(f"Error type: {type(e).__name__}")
            raise

        # Print results
        if verbose:
            print(f"\n{'='*80}")
            print(" "*25 + "TUNING RESULTS")
            print(f"{'='*80}")
            print(f"\nBest parameters found:")
            for param, value in search_cv.best_params_.items():
                print(f"  {param:25s}: {value}")
            print(f"\nBest CV {scoring} score: {search_cv.best_score_:.5f}")
            print("="*80 + "\n")

        return search_cv.best_estimator_, search_cv.best_params_, search_cv.best_score_

    def tune_classification_models(self, X_train, y_train, models=None,
                                   search_type='grid', cv=5):
        """
        Tune multiple classification models

        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model names to tune (default: all)
            search_type: 'grid' or 'random'
            cv: Number of CV folds

        Returns:
            dict: Dictionary mapping model names to (best_model, best_params, best_score)

        Example:
            >>> tuner = ModelTuner()
            >>> results = tuner.tune_classification_models(
            ...     X_train, y_train,
            ...     models=['RandomForest', 'SVC']
            ... )
        """
        if models is None:
            models = ['RandomForest', 'DecisionTree', 'SVC', 'KNN']

        results = {}
        for model_name in models:
            best_model, best_params, best_score = self.tune_model(
                X_train, y_train,
                model_type='classification',
                model_name=model_name,
                search_type=search_type,
                cv=cv
            )
            results[model_name] = (best_model, best_params, best_score)

        return results

    def tune_regression_models(self, X_train, y_train, models=None,
                               search_type='grid', cv=5):
        """
        Tune multiple regression models

        Args:
            X_train: Training features
            y_train: Training values
            models: List of model names to tune (default: all)
            search_type: 'grid' or 'random'
            cv: Number of CV folds

        Returns:
            dict: Dictionary mapping model names to (best_model, best_params, best_score)

        Example:
            >>> tuner = ModelTuner()
            >>> results = tuner.tune_regression_models(
            ...     X_train, y_train,
            ...     models=['RandomForest', 'SVM']
            ... )
        """
        if models is None:
            models = ['RandomForest', 'DecisionTree', 'SVM', 'KNN']

        results = {}
        for model_name in models:
            best_model, best_params, best_score = self.tune_model(
                X_train, y_train,
                model_type='regression',
                model_name=model_name,
                search_type=search_type,
                cv=cv
            )
            results[model_name] = (best_model, best_params, best_score)

        return results

    def _get_model_and_params(self, model_type, model_name, param_grid):
        """
        Get base model and parameter grid

        Args:
            model_type: 'classification' or 'regression'
            model_name: Model name
            param_grid: Custom param grid or None

        Returns:
            tuple: (base_model, param_grid)
        """
        # Get parameter grid (check overrides first, then fall back to defaults)
        using_override = False
        if param_grid is None:
            if model_type == 'classification':
                # Check for override in hyperparameters.py first
                if model_name in CLF_OVERRIDES:
                    param_grid = CLF_OVERRIDES[model_name]
                    using_override = True
                else:
                    param_grid = self.CLF_PARAM_GRIDS.get(model_name)
            else:  # regression
                # Map 'SVC' to 'SVM' for regression
                grid_name = 'SVM' if model_name == 'SVC' else model_name
                # Check for override in hyperparameters.py first
                if grid_name in REG_OVERRIDES:
                    param_grid = REG_OVERRIDES[grid_name]
                    using_override = True
                else:
                    param_grid = self.REG_PARAM_GRIDS.get(grid_name)

        # Store override status for verbose output
        self._last_override_used = using_override

        if param_grid is None:
            raise ValueError(f"No parameter grid defined for {model_name} ({model_type})")

        # Get base model
        if model_type == 'classification':
            if model_name == 'RandomForest':
                base_model = RandomForestClassifier(random_state=self.random_state)
            elif model_name == 'DecisionTree':
                base_model = DecisionTreeClassifier(random_state=self.random_state)
            elif model_name == 'SVC':
                base_model = svm.SVC(probability=True, random_state=self.random_state)
            elif model_name == 'KNN':
                base_model = neighbors.KNeighborsClassifier()
            elif model_name == 'LogisticRegression':
                base_model = LogisticRegression(random_state=self.random_state)
            elif model_name == 'NaiveBayes':
                base_model = GaussianNB()
            elif model_name == 'Bagging':
                base_model = BaggingClassifier(
                    estimator=DecisionTreeClassifier(),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'Stacking':
                estimators = [
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                    ('dt', DecisionTreeClassifier(random_state=self.random_state)),
                    ('svc', svm.SVC(probability=True, random_state=self.random_state)),
                    ('knn', neighbors.KNeighborsClassifier()),
                    ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state))
                ]
                base_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported classification model: {model_name}")
        else:  # regression
            if model_name == 'RandomForest':
                base_model = RandomForestRegressor(random_state=self.random_state)
            elif model_name == 'DecisionTree':
                base_model = DecisionTreeRegressor(random_state=self.random_state)
            elif model_name == 'SVM':
                base_model = svm.SVR()
            elif model_name == 'KNN':
                base_model = neighbors.KNeighborsRegressor()
            elif model_name == 'LinearRegression':
                base_model = LinearRegression()
            elif model_name == 'Ridge':
                base_model = Ridge(random_state=self.random_state)
            elif model_name == 'Lasso':
                base_model = Lasso(random_state=self.random_state)
            elif model_name == 'ElasticNet':
                base_model = ElasticNet(random_state=self.random_state)
            elif model_name == 'GradientBoosting':
                base_model = GradientBoostingRegressor(random_state=self.random_state)
            elif model_name == 'XGBoost':
                base_model = xgb.XGBRegressor(random_state=self.random_state)
            elif model_name == 'LightGBM':
                base_model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            elif model_name == 'Bagging':
                base_model = BaggingRegressor(
                    estimator=DecisionTreeRegressor(),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_name == 'Stacking':
                estimators = [
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                    ('dt', DecisionTreeRegressor(random_state=self.random_state)),
                    ('svr', svm.SVR()),
                    ('knn', neighbors.KNeighborsRegressor()),
                    ('ridge', Ridge(random_state=self.random_state)),
                    ('lasso', Lasso(random_state=self.random_state))
                ]
                base_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(random_state=self.random_state),
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported regression model: {model_name}")

        return base_model, param_grid

    def _count_combinations(self, param_grid):
        """
        Count total number of parameter combinations

        Args:
            param_grid: Either a dict or a list of dicts

        Returns:
            Total number of combinations
        """
        # Handle list of parameter grids (different estimators as dicts)
        if isinstance(param_grid, list):
            total_count = 0
            for single_grid in param_grid:
                count = 1
                for values in single_grid.values():
                    count *= len(values)
                total_count += count
            return total_count

        # Handle single parameter grid dictionary
        else:
            count = 1
            for values in param_grid.values():
                count *= len(values)
            return count

