"""
Multi-Output Regression Module

This module handles training models that predict multiple targets simultaneously,
specifically for predicting both heating_load and cooling_load.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm, neighbors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb


class MultiOutputPredictor:
    """
    Handles multi-output regression for predicting heating_load and cooling_load simultaneously

    This approach is superior to predicting aggregated values because it:
    - Preserves information about individual components
    - Learns correlations between the two outputs
    - Allows derivation of sum/classes from individual predictions
    """

    def __init__(self, random_state=42):
        """
        Initialize the multi-output predictor

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.target_names = ['heating_load', 'cooling_load']

    def train_random_forest(self, X_train, X_test, y_train, y_test,
                           n_estimators=100, max_depth=None, verbose=True):
        """
        Train Random Forest for multi-output regression

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets (must be 2D with heating_load and cooling_load)
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            verbose: Whether to print results

        Returns:
            tuple: (model, metrics_dict)
        """
        if verbose:
            print("="*80)
            print(" "*20 + "MULTI-OUTPUT RANDOM FOREST")
            print("="*80)

        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics for each output
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['RandomForest'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_decision_tree(self, X_train, X_test, y_train, y_test,
                           max_depth=None, verbose=True):
        """
        Train Decision Tree for multi-output regression
        """
        if verbose:
            print("="*80)
            print(" "*20 + "MULTI-OUTPUT DECISION TREE")
            print("="*80)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['DecisionTree'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_knn(self, X_train, X_test, y_train, y_test,
                  n_neighbors=5, verbose=True):
        """
        Train KNN for multi-output regression
        """
        if verbose:
            print("="*80)
            print(" "*20 + "MULTI-OUTPUT KNN")
            print("="*80)

        model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['KNN'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_linear_regression(self, X_train, X_test, y_train, y_test, verbose=True):
        """
        Train Linear Regression for multi-output regression
        """
        if verbose:
            print("="*80)
            print(" "*18 + "MULTI-OUTPUT LINEAR REGRESSION")
            print("="*80)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['LinearRegression'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_ridge_regression(self, X_train, X_test, y_train, y_test,
                               alpha=1.0, verbose=True):
        """
        Train Ridge Regression for multi-output regression
        """
        if verbose:
            print("="*80)
            print(" "*18 + "MULTI-OUTPUT RIDGE REGRESSION")
            print("="*80)

        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['Ridge'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_lasso_regression(self, X_train, X_test, y_train, y_test,
                               alpha=1.0, verbose=True):
        """
        Train Lasso Regression for multi-output regression using MultiOutputRegressor
        Note: Lasso doesn't natively support multi-output, so we wrap it
        """
        if verbose:
            print("="*80)
            print(" "*18 + "MULTI-OUTPUT LASSO REGRESSION")
            print("="*80)

        base_model = Lasso(alpha=alpha, random_state=self.random_state)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['Lasso'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_elasticnet_regression(self, X_train, X_test, y_train, y_test,
                                    alpha=1.0, l1_ratio=0.5, verbose=True):
        """
        Train ElasticNet Regression for multi-output regression using MultiOutputRegressor
        Note: ElasticNet doesn't natively support multi-output, so we wrap it
        """
        if verbose:
            print("="*80)
            print(" "*17 + "MULTI-OUTPUT ELASTICNET REGRESSION")
            print("="*80)

        base_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['ElasticNet'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test,
                                n_estimators=100, learning_rate=0.1, max_depth=3,
                                verbose=True):
        """
        Train Gradient Boosting for multi-output regression using MultiOutputRegressor
        Note: GradientBoostingRegressor doesn't natively support multi-output, so we wrap it
        """
        if verbose:
            print("="*80)
            print(" "*13 + "MULTI-OUTPUT GRADIENT BOOSTING REGRESSION")
            print("="*80)

        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['GradientBoosting'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_xgboost(self, X_train, X_test, y_train, y_test,
                      n_estimators=100, learning_rate=0.1, max_depth=3,
                      verbose=True):
       
        if verbose:
            print("="*80)
            print(" "*18 + "MULTI-OUTPUT XGBOOST REGRESSION")
            print("="*80)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['XGBoost'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def train_lightgbm(self, X_train, X_test, y_train, y_test,
                       n_estimators=100, learning_rate=0.1, max_depth=3,
                       verbose=True):
        """
        Train LightGBM for multi-output regression
        LightGBM natively supports multi-output regression
        """
       
        if verbose:
            print("="*80)
            print(" "*17 + "MULTI-OUTPUT LIGHTGBM REGRESSION")
            print("="*80)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            verbose=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, verbose)

        self.models['LightGBM'] = model

        if verbose:
            print("="*80 + "\n")

        return model, metrics

    def tune_random_forest(self, X_train, y_train, cv=5, verbose=True):
        """
        Tune Random Forest hyperparameters for multi-output regression

        Args:
            X_train: Training features
            y_train: Training targets (2D)
            cv: Cross-validation folds
            verbose: Whether to print results

        Returns:
            tuple: (best_model, best_params, best_score)
        """
        if verbose:
            print("="*80)
            print(" "*15 + "TUNING MULTI-OUTPUT RANDOM FOREST")
            print("="*80)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        if verbose:
            print(f"\nTuning with {cv}-fold cross-validation...")
            print(f"Parameter combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])}")

        grid_search.fit(X_train, y_train)

        if verbose:
            print(f"\nBest parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"  {param:20s}: {value}")
            print(f"\nBest CV R² score: {grid_search.best_score_:.5f}")
            print("="*80 + "\n")

        self.models['RandomForest_tuned'] = grid_search.best_estimator_

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def predict(self, model, X_new):
        """
        Make predictions with a trained multi-output model

        Args:
            model: Trained model
            X_new: New features

        Returns:
            dict: Dictionary with predictions for each output
        """
        predictions = model.predict(X_new)

        return {
            'heating_load': predictions[:, 0],
            'cooling_load': predictions[:, 1],
            'charges_sum': predictions[:, 0] + predictions[:, 1],
            'predictions_array': predictions
        }

    def predict_and_classify(self, model, X_new, class_thresholds):
        """
        Predict both outputs and derive classification

        Args:
            model: Trained model
            X_new: New features
            class_thresholds: Thresholds for classification (e.g., [30, 40, 50])

        Returns:
            dict: Predictions with derived classification
        """
        result = self.predict(model, X_new)

        # Classify based on sum
        charges_sum = result['charges_sum']
        classes = np.digitize(charges_sum, class_thresholds)
        result['charges_classes'] = classes

        return result

    def _calculate_metrics(self, y_true, y_pred, verbose=True):
        """
        Calculate metrics for each output separately and combined

        Args:
            y_true: True values (n_samples, 2)
            y_pred: Predicted values (n_samples, 2)
            verbose: Whether to print metrics

        Returns:
            dict: Metrics for each output
        """
        metrics = {}

        for i, target_name in enumerate(self.target_names):
            y_true_i = y_true[:, i] if y_true.ndim > 1 else y_true
            y_pred_i = y_pred[:, i] if y_pred.ndim > 1 else y_pred

            r2 = r2_score(y_true_i, y_pred_i)
            rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
            mae = mean_absolute_error(y_true_i, y_pred_i)
            mse = mean_squared_error(y_true_i, y_pred_i)

            metrics[target_name] = {
                'R2_Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            }

            if verbose:
                print(f"\nMetrics for {target_name}:")
                print(f"  R² Score: {r2:.5f}")
                print(f"  RMSE:     {rmse:.5f}")
                print(f"  MAE:      {mae:.5f}")
                print(f"  MSE:      {mse:.5f}")

        # Overall metrics (averaged)
        if verbose:
            avg_r2 = np.mean([metrics[t]['R2_Score'] for t in self.target_names])
            avg_rmse = np.mean([metrics[t]['RMSE'] for t in self.target_names])
            print(f"\nOverall (averaged):")
            print(f"  Avg R² Score: {avg_r2:.5f}")
            print(f"  Avg RMSE:     {avg_rmse:.5f}")

        metrics['overall'] = {
            'Avg_R2': np.mean([metrics[t]['R2_Score'] for t in self.target_names]),
            'Avg_RMSE': np.mean([metrics[t]['RMSE'] for t in self.target_names])
        }

        return metrics

    def compare_with_aggregated(self, y_test_multi, y_pred_multi,
                               y_test_agg, y_pred_agg, verbose=True):
        """
        Compare multi-output predictions with aggregated approach

        Args:
            y_test_multi: True values for multi-output (n_samples, 2)
            y_pred_multi: Predictions from multi-output model (n_samples, 2)
            y_test_agg: True aggregated values (n_samples,)
            y_pred_agg: Predictions from aggregated model (n_samples,)
            verbose: Whether to print comparison

        Returns:
            dict: Comparison metrics
        """
        # Sum from multi-output predictions
        sum_from_multi = y_pred_multi[:, 0] + y_pred_multi[:, 1]
        true_sum = y_test_multi[:, 0] + y_test_multi[:, 1]

        # Metrics for multi-output sum
        r2_multi = r2_score(true_sum, sum_from_multi)
        rmse_multi = np.sqrt(mean_squared_error(true_sum, sum_from_multi))

        # Metrics for direct aggregated prediction
        r2_agg = r2_score(y_test_agg, y_pred_agg)
        rmse_agg = np.sqrt(mean_squared_error(y_test_agg, y_pred_agg))

        if verbose:
            print("="*80)
            print(" "*20 + "APPROACH COMPARISON")
            print("="*80)
            print("\nPredicting SUM (aggregated target):")
            print(f"  Multi-output approach (sum of components):")
            print(f"    R² Score: {r2_multi:.5f}")
            print(f"    RMSE:     {rmse_multi:.5f}")
            print(f"\n  Direct aggregated approach:")
            print(f"    R² Score: {r2_agg:.5f}")
            print(f"    RMSE:     {rmse_agg:.5f}")
            print(f"\n  Difference:")
            print(f"    ΔR²:      {r2_multi - r2_agg:+.5f}")
            print(f"    ΔRMSE:    {rmse_multi - rmse_agg:+.5f}")
            print("="*80 + "\n")

        return {
            'multi_output_sum': {'R2': r2_multi, 'RMSE': rmse_multi},
            'aggregated': {'R2': r2_agg, 'RMSE': rmse_agg},
            'advantage': {
                'R2_improvement': r2_multi - r2_agg,
                'RMSE_improvement': rmse_agg - rmse_multi  # Lower is better for RMSE
            }
        }


def create_multioutput_targets(df, *target_cols):
    """
    Create multi-output target array from dataframe

    Args:
        df: DataFrame containing target columns
        *target_cols: Variable number of column names (minimum 2).
            If no columns provided, defaults to ('heating_load', 'cooling_load')

    Returns:
        numpy.ndarray: Array of shape (n_samples, n_targets)

    Examples:
        # Default usage (backward compatible)
        y = create_multioutput_targets(df, 'target1', 'target2')

        # Multiple targets
        y = create_multioutput_targets(df, 'target1', 'target2', 'target3', 'target4')
    """
    if not target_cols:
        target_cols = ('target1', 'target2')  # Default target columns

    if len(target_cols) < 2:
        raise ValueError("At least 2 target columns are required for multi-output regression")

    return df[list(target_cols)].values
