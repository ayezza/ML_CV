"""
Model Training Module

This module handles training of machine learning models for classification and regression.
"""
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, VotingClassifier,
                              GradientBoostingRegressor, BaggingClassifier, BaggingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm, neighbors, metrics
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support
from core.metrics import MetricsCollector
from visualization.classification import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_classification_bar_chart,
    plot_probability_matrix
)

from visualization.learning_curves_summary import create_learning_curves_summary_table
from visualization.regression import plot_regression_scatter
from visualization.analysis import plot_learning_curve
import pandas as pd


class ModelTrainer:
    """
    Handles training and evaluation of ML models

    This class provides methods to train different models for classification
    and regression tasks, along with automatic visualization and metrics collection.
    """

    def __init__(self, metrics_collector=None, output_config=None):
        """
        Initialize the model trainer

        Args:
            metrics_collector: MetricsCollector instance (optional)
            output_config: Config instance for output paths (optional)
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.output_config = output_config
        self.trained_models = {}
        self.learning_curve_stats = []  # Collect learning curve statistics for summary table

    def _save_predictions(self, y_test, y_pred, y_score, model_name, model_type='classification', is_tuned=False):
        """
        Save predictions to CSV file

        Args:
            y_test: True labels/values
            y_pred: Predicted labels/values
            y_score: Prediction probabilities (classification) or None (regression)
            model_name: Name of the model
            model_type: 'classification' or 'regression'
            is_tuned: Whether this is a tuned model

        Returns:
            Path to saved CSV file, or None if output_config not available
        """
        if not self.output_config:
            return None

        # Determine output directory
        if model_type == 'classification':
            output_dir = self.output_config.CLF_PREDICTIONS_DIR
        else:  # regression
            output_dir = self.output_config.REG_PREDICTIONS_DIR

        # Ensure directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create base DataFrame
        data = {
            'y_true': y_test,
            'y_pred': y_pred
        }

        # Add probability scores for classification
        if model_type == 'classification' and y_score is not None:
            # Add probability for each class
            n_classes = y_score.shape[1] if len(y_score.shape) > 1 else 1
            for class_idx in range(n_classes):
                if len(y_score.shape) > 1:
                    data[f'prob_class_{class_idx}'] = y_score[:, class_idx]
                else:
                    data[f'prob_class_0'] = y_score

        # Create DataFrame
        df = pd.DataFrame(data)

        # Add index as sample ID
        df.insert(0, 'sample_id', range(len(df)))

        # Generate filename
        tuned_suffix = '_tuned' if is_tuned else '_base'
        filename = f'{model_name}{tuned_suffix}_predictions.csv'
        filepath = output_dir / filename

        # Save to CSV
        df.to_csv(filepath, index=False)

        return filepath

    def _generate_learning_curve(self, model, X_train, y_train, model_name, model_type='classification', search_type='grid'):
        """
        Generate learning curve for a trained model (internal helper method)

        Args:
            model: Trained sklearn model
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            model_type: 'classification' or 'regression'
            search_type: 'grid' or 'random' for hyperparameter tuning method
        """
        if not self.output_config:
            return

        # Check if learning curves are enabled in config
        if not getattr(self.output_config, 'GENERATE_LEARNING_CURVES', False):
            return

        # Determine output directory based on model type
        if model_type == 'classification':
            output_path = self.output_config.CLF_LEARNING_CURVES_DIR
        else:  # regression
            output_path = self.output_config.REG_LEARNING_CURVES_DIR

        # Get train sizes from config or use default
        train_sizes = getattr(self.output_config, 'LEARNING_CURVE_TRAIN_SIZES',
                             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Get CV folds from config
        cv = getattr(self.output_config, 'CV_FOLDS', 5)

        # Get n_jobs from config
        n_jobs = getattr(self.output_config, 'N_JOBS', -1)

        # Generate learning curve
        try:
            save_path, stats = plot_learning_curve(
                estimator=model,
                X=X_train,
                y=y_train,
                model_name=model_name,
                model_type=model_type,
                output_path=output_path,
                cv=cv,
                n_jobs=n_jobs,
                train_sizes=train_sizes,
                scoring=None,  # Use default scoring
                figsize=(10, 6),
                search_type=search_type  # Pass search_type for filename and labeling
            )
            # Collect stats for summary table
            if stats:
                self.learning_curve_stats.append(stats)
        except Exception as e:
            print(f"[WARNING] Could not generate learning curve for {model_name}: {e}")

    def generate_learning_curves_summary(self):
        """
        Generate summary table of all learning curve statistics.

        Returns:
            Path to saved summary table image, or None if no stats available
        """
        if not self.output_config:
            print("[WARNING] No output config provided - cannot save summary table")
            return None

        if not self.learning_curve_stats:
            print("[WARNING] No learning curve statistics collected - run models with GENERATE_LEARNING_CURVES=True")
            return None

        # Use the base learning curves directory
        output_path = getattr(self.output_config, 'LEARNING_CURVES_DIR', None)

        return create_learning_curves_summary_table(self.learning_curve_stats, output_path)
    

    def train_random_forest_classifier(self, X_train, X_test, y_train, y_test,
                                       n_estimators=100, random_state=42,
                                       is_tuned=False, custom_aggregation_name='sum',
                                       visualize=True, search_type='grid'):
        """
        Train Random Forest Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            n_estimators: Number of trees (default: 100)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling (default: sum)
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Random Forest model

        Example:
            >>> trainer = ModelTrainer()
            >>> model = trainer.train_random_forest_classifier(
            ...     X_train, X_test, y_train, y_test
            ... )
        """
        print("="*80)
        print(" "*20 + "RANDOM FOREST CLASSIFICATION")
        print("="*80)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('RandomForest', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ Random Forest Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'RandomForest Classification', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'RandomForest', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'RandomForest', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'RandomForest', 'classification', search_type)

        print("="*80 + "\n")
        self.trained_models['RandomForest_clf'] = model
        return model

    def train_decision_tree_classifier(self, X_train, X_test, y_train, y_test,
                                       random_state=42, is_tuned=False,
                                       custom_aggregation_name='sum', visualize=True):
        """
        Train Decision Tree Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Decision Tree model
        """
        print("="*80)
        print(" "*20 + "DECISION TREE CLASSIFICATION")
        print("="*80)

        # Train model
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('DecisionTree', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ Decision Tree Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'DecisionTree Classification', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'DecisionTree', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'DecisionTree', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'DecisionTree', 'classification')

        print("="*80 + "\n")
        self.trained_models['DecisionTree_clf'] = model
        return model

    def train_svc(self, X_train, X_test, y_train, y_test,
                 kernel='rbf', random_state=42, is_tuned=False,
                 custom_aggregation_name='sum', visualize=True):
        """
        Train Support Vector Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            kernel: Kernel type (default: 'rbf')
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained SVC model
        """
        print("="*80)
        print(" "*25 + "SVC CLASSIFICATION")
        print("="*80)

        # Train model
        model = svm.SVC(kernel=kernel, probability=True, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('SVC', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ SVC Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'SVC Classification', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'SVC', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'SVC', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'SVC', 'classification')

        print("="*80 + "\n")
        self.trained_models['SVC'] = model
        return model

    def train_knn_classifier(self, X_train, X_test, y_train, y_test,
                            n_neighbors=5, is_tuned=False,
                            custom_aggregation_name='sum', visualize=True):
        """
        Train K-Nearest Neighbors Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            n_neighbors: Number of neighbors (default: 5)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained KNN model
        """
        print("="*80)
        print(" "*25 + "KNN CLASSIFICATION")
        print("="*80)

        # Train model
        model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('KNN', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ KNN Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'KNN Classification', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'KNN', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'KNN', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'KNN', 'classification')

        print("="*80 + "\n")
        self.trained_models['KNN_clf'] = model
        return model

    def train_logistic_regression_classifier(self, X_train, X_test, y_train, y_test,
                                              max_iter=1000, random_state=42,
                                              is_tuned=False, custom_aggregation_name='sum',
                                              visualize=True):
        """
        Train Logistic Regression Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            max_iter: Maximum iterations for convergence (default: 1000)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Logistic Regression model
        """
        print("="*80)
        print(" "*18 + "LOGISTIC REGRESSION CLASSIFICATION")
        print("="*80)

        # Train model
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('LogisticRegression', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ Logistic Regression Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Logistic Regression Classification', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'LogisticRegression', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'LogisticRegression', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'LogisticRegression', 'classification')

        print("="*80 + "\n")
        self.trained_models['LogisticRegression_clf'] = model
        return model

    def train_naive_bayes_classifier(self, X_train, X_test, y_train, y_test,
                                      is_tuned=False, custom_aggregation_name='sum',
                                      visualize=True):
        """
        Train Gaussian Naive Bayes Classifier

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Naive Bayes model
        """
        print("="*80)
        print(" "*22 + "NAIVE BAYES CLASSIFICATION")
        print("="*80)

        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('NaiveBayes', 'classification', is_tuned, metrics_dict)
        print(f"\n✓ Naive Bayes Accuracy: {metrics_dict['Accuracy']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Naive Bayes Classification', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'NaiveBayes', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'NaiveBayes', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'NaiveBayes', 'classification')

        print("="*80 + "\n")
        self.trained_models['NaiveBayes_clf'] = model
        return model

    def train_voting_classifier(self, X_train, X_test, y_train, y_test,
                                voting='hard', is_tuned=False,
                                custom_aggregation_name='sum', visualize=True):
        """
        Train Voting Classifier (ensemble of RF, SVC, KNN)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            voting: 'hard' or 'soft' voting (default: 'hard')
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Voting Classifier model
        """
        print("="*80)
        print(" "*20 + f"VOTING CLASSIFIER ({voting.upper()})")
        print("="*80)

        # Create base estimators
        clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
        clf2 = svm.SVC(kernel='rbf', probability=True, random_state=42)
        clf3 = neighbors.KNeighborsClassifier(n_neighbors=5)

        # Create voting classifier
        model = VotingClassifier(
            estimators=[('rf', clf1), ('svc', clf2), ('knn', clf3)],
            voting=voting
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        if voting == 'soft':
            y_score = model.predict_proba(X_test)
            metrics_dict = self.metrics_collector.collect_classification_metrics(
                y_test, y_pred, y_score
            )
        else:
            # Hard voting - no probabilities
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            metrics_dict = {
                'Accuracy': accuracy,
                'Precision_Macro': precision,
                'Recall_Macro': recall,
                'F1_Macro': f1,
                'AUC_ROC_Micro': None
            }

        self.metrics_collector.add_metrics(
            f'Voting_{voting.capitalize()}', 'classification', is_tuned, metrics_dict
        )
        print(f"\n✓ Voting Classifier Accuracy: {metrics_dict['Accuracy']:.5f} ({voting} voting)")

        # Save predictions to CSV
        y_score_to_save = y_score if voting == 'soft' else None
        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score_to_save, f'Voting_{voting.capitalize()}', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            model_name = f'Voting_{voting.capitalize()}'
            output_base = self.output_config.CLF_TUNED_DIR if is_tuned else self.output_config.CLF_BASE_DIR

            # Confusion matrix
            plot_confusion_matrix(
                y_test, y_pred,
                target_classes=sorted(y_test.unique()),
                model_name=model_name,
                output_path=output_base / 'confusion_matrices',
                is_tuned=is_tuned,
                custom_aggregation_name=custom_aggregation_name
            )

            # ROC curve (only for soft voting)
            if voting == 'soft':
                plot_roc_curve(
                    y_test, y_score,
                    model_name=model_name,
                    output_path=output_base / 'roc_curves',
                    is_tuned=is_tuned,
                    custom_aggregation_name=custom_aggregation_name
                )

            # Bar chart
            plot_classification_bar_chart(
                y_test, y_pred,
                model_name=model_name,
                output_path=output_base / 'bar_charts',
                is_tuned=is_tuned,
                custom_aggregation_name=custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Voting', 'classification')

        print("="*80 + "\n")
        self.trained_models[f'Voting_{voting}'] = model
        return model

    def train_random_forest_regressor(self, X_train, X_test, y_train, y_test,
                                      n_estimators=100, random_state=42,
                                      is_tuned=False, custom_aggregation_name='sum',
                                      visualize=True):
        """
        Train Random Forest Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            n_estimators: Number of trees (default: 100)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Random Forest Regressor model
        """
        print("="*80)
        print(" "*20 + "RANDOM FOREST REGRESSION")
        print("="*80)

        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('RandomForest', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Random Forest - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'RandomForest Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'RandomForest', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'RandomForest_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'RandomForest', 'regression')

        print("="*80 + "\n")
        self.trained_models['RandomForest_reg'] = model
        return model

    def train_decision_tree_regressor(self, X_train, X_test, y_train, y_test,
                                      random_state=42, is_tuned=False,
                                      custom_aggregation_name='sum', visualize=True):
        """
        Train Decision Tree Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Decision Tree Regressor model
        """
        print("="*80)
        print(" "*20 + "DECISION TREE REGRESSION")
        print("="*80)

        # Train model
        model = DecisionTreeRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('DecisionTree', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Decision Tree - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'DecisionTree Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'DecisionTree', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'DecisionTree_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'DecisionTree', 'regression')

        print("="*80 + "\n")
        self.trained_models['DecisionTree_reg'] = model
        return model

    def train_svm_regressor(self, X_train, X_test, y_train, y_test,
                           kernel='rbf', is_tuned=False,
                           custom_aggregation_name='sum', visualize=True):
        """
        Train Support Vector Machine Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            kernel: Kernel type (default: 'rbf')
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained SVR model
        """
        print("="*80)
        print(" "*25 + "SVM REGRESSION")
        print("="*80)

        # Train model with improved default parameters to avoid underfitting
        model = svm.SVR(
            kernel=kernel,
            C=100.0,          # Higher C = less regularization (allows more complex model)
            gamma='scale',    # Auto-scaled based on features
            epsilon=0.1       # Standard epsilon-tube width
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('SVM', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ SVM - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'SVM Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'SVM', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'SVM_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'SVR', 'regression')

        print("="*80 + "\n")
        self.trained_models['SVM_reg'] = model
        return model

    def train_knn_regressor(self, X_train, X_test, y_train, y_test,
                           n_neighbors=5, is_tuned=False,
                           custom_aggregation_name='sum', visualize=True):
        """
        Train K-Nearest Neighbors Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            n_neighbors: Number of neighbors (default: 5)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained KNN Regressor model
        """
        print("="*80)
        print(" "*25 + "KNN REGRESSION")
        print("="*80)

        # Train model
        model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('KNN', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ KNN - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'KNN Regression', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'KNN', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'KNN_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'KNN', 'regression')

        print("="*80 + "\n")
        self.trained_models['KNN_reg'] = model
        return model

    def train_linear_regression(self, X_train, X_test, y_train, y_test,
                                is_tuned=False, custom_aggregation_name='sum',
                                visualize=True):
        """
        Train Linear Regression

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Linear Regression model
        """
        print("="*80)
        print(" "*23 + "LINEAR REGRESSION")
        print("="*80)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('LinearRegression', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Linear Regression - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Linear Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'LinearRegression', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'LinearRegression_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Linear', 'regression')

        print("="*80 + "\n")
        self.trained_models['LinearRegression_reg'] = model
        return model

    def train_ridge_regression(self, X_train, X_test, y_train, y_test,
                               alpha=1.0, random_state=42, is_tuned=False,
                               custom_aggregation_name='sum', visualize=True):
        """
        Train Ridge Regression (L2 regularization)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            alpha: Regularization strength (default: 1.0)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Ridge Regression model
        """
        print("="*80)
        print(" "*23 + "RIDGE REGRESSION")
        print("="*80)

        # Train model
        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('Ridge', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Ridge Regression - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Ridge Regression', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'Ridge', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'Ridge_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Ridge', 'regression')

        print("="*80 + "\n")
        self.trained_models['Ridge_reg'] = model
        return model

    def train_lasso_regression(self, X_train, X_test, y_train, y_test,
                               alpha=1.0, random_state=42, is_tuned=False,
                               custom_aggregation_name='sum', visualize=True):
        """
        Train Lasso Regression (L1 regularization)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            alpha: Regularization strength (default: 1.0)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Lasso Regression model
        """
        print("="*80)
        print(" "*23 + "LASSO REGRESSION")
        print("="*80)

        # Train model
        model = Lasso(alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('Lasso', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Lasso Regression - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Lasso Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'Lasso', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'Lasso_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Lasso', 'regression')

        print("="*80 + "\n")
        self.trained_models['Lasso_reg'] = model
        return model

    def train_elasticnet_regression(self, X_train, X_test, y_train, y_test,
                                    alpha=1.0, l1_ratio=0.5, random_state=42,
                                    is_tuned=False, custom_aggregation_name='sum',
                                    visualize=True):
        """
        Train ElasticNet Regression (L1 + L2 regularization)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            alpha: Regularization strength (default: 1.0)
            l1_ratio: Mix of L1 and L2 (0=L2, 1=L1, default: 0.5)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained ElasticNet model
        """
        print("="*80)
        print(" "*22 + "ELASTICNET REGRESSION")
        print("="*80)

        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('ElasticNet', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ ElasticNet - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'ElasticNet Regression', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'ElasticNet', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'ElasticNet_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'ElasticNet', 'regression')

        print("="*80 + "\n")
        self.trained_models['ElasticNet_reg'] = model
        return model

    def train_gradient_boosting_regressor(self, X_train, X_test, y_train, y_test,
                                         n_estimators=100, learning_rate=0.1,
                                         max_depth=3, random_state=42, is_tuned=False,
                                         custom_aggregation_name='sum', visualize=True):
        """
        Train Gradient Boosting Regressor (sklearn implementation)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            n_estimators: Number of boosting stages (default: 100)
            learning_rate: Learning rate (default: 0.1)
            max_depth: Maximum depth of trees (default: 3)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained Gradient Boosting model
        """
        print("="*80)
        print(" "*18 + "GRADIENT BOOSTING REGRESSION")
        print("="*80)

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('GradientBoosting', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Gradient Boosting - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'Gradient Boosting Regression', is_tuned)

       # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'GradientBoosting', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'GradientBoosting_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'GradientBoosting', 'regression')

        print("="*80 + "\n")
        self.trained_models['GradientBoosting_reg'] = model
        return model

    def train_xgboost_regressor(self, X_train, X_test, y_train, y_test,
                                n_estimators=100, learning_rate=0.1, max_depth=3,
                                random_state=42, is_tuned=False,
                                custom_aggregation_name='sum', visualize=True):
        """
        Train XGBoost Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            n_estimators: Number of boosting rounds (default: 100)
            learning_rate: Learning rate (default: 0.1)
            max_depth: Maximum depth of trees (default: 3)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained XGBoost model

        Note:
            Requires xgboost library: pip install xgboost
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            )

        print("="*80)
        print(" "*23 + "XGBOOST REGRESSION")
        print("="*80)

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('XGBoost', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ XGBoost - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'XGBoost Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'XGBoost', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'XGBoost_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'XGBoost', 'regression')

        print("="*80 + "\n")
        self.trained_models['XGBoost_reg'] = model
        return model

    def train_lightgbm_regressor(self, X_train, X_test, y_train, y_test,
                                 n_estimators=100, learning_rate=0.1, max_depth=3,
                                 random_state=42, is_tuned=False,
                                 custom_aggregation_name='sum', visualize=True):
        """
        Train LightGBM Regressor

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test values
            n_estimators: Number of boosting rounds (default: 100)
            learning_rate: Learning rate (default: 0.1)
            max_depth: Maximum depth of trees (default: 3)
            random_state: Random seed (default: 42)
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Aggregation function name for labeling
            visualize: Whether to generate visualizations

        Returns:
            model: Trained LightGBM model

        Note:
            Requires lightgbm library: pip install lightgbm
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            )

        print("="*80)
        print(" "*22 + "LIGHTGBM REGRESSION")
        print("="*80)

        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbose=-1  # Suppress warnings
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('LightGBM', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ LightGBM - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Print parameters
        self._print_model_parameters(model, 'LightGBM Regression', is_tuned)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'LightGBM', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'LightGBM_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'LightGBM', 'regression')

        print("="*80 + "\n")
        self.trained_models['LightGBM_reg'] = model
        return model

    # ========================================================================
    # ENSEMBLE METHODS - BAGGING
    # ========================================================================

    def train_bagging_classifier(self, X_train, X_test, y_train, y_test,
                                  n_estimators=50, max_samples=0.8, max_features=0.8,
                                  random_state=42, is_tuned=False,
                                  custom_aggregation_name='sum', visualize=True):
        """
        Train Bagging Classifier (ensemble of Decision Trees)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            n_estimators: Number of base estimators
            max_samples: Proportion of samples for each estimator
            max_features: Proportion of features for each estimator
            random_state: Random seed
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Name of aggregation used
            visualize: Whether to generate visualizations

        Returns:
            Trained BaggingClassifier model
        """
        print("="*80)
        print(" "*25 + "BAGGING CLASSIFIER")
        print("="*80)

        # Train model
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        print(f"\nBagging Classifier Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1_score:.4f}")

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('Bagging', 'classification', is_tuned, metrics_dict)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'Bagging', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'Bagging_Classifier', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Bagging', 'classification')

        print("="*80 + "\n")
        self.trained_models['Bagging_clf'] = model
        return model

    def train_bagging_regressor(self, X_train, X_test, y_train, y_test,
                                n_estimators=50, max_samples=0.8, max_features=0.8,
                                random_state=42, is_tuned=False,
                                custom_aggregation_name='sum', visualize=True):
        """
        Train Bagging Regressor (ensemble of Decision Trees)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            n_estimators: Number of base estimators
            max_samples: Proportion of samples for each estimator
            max_features: Proportion of features for each estimator
            random_state: Random seed
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Name of aggregation used
            visualize: Whether to generate visualizations

        Returns:
            Trained BaggingRegressor model
        """
        print("="*80)
        print(" "*25 + "BAGGING REGRESSOR")
        print("="*80)

        # Train model
        model = BaggingRegressor(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = metrics.r2_score(y_test, y_pred)

        print(f"\nBagging Regressor Performance:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('Bagging', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Bagging - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'Bagging', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'Bagging_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Bagging', 'regression')

        print("="*80 + "\n")
        self.trained_models['Bagging_reg'] = model
        return model

    # ========================================================================
    # ENSEMBLE METHODS - STACKING
    # ========================================================================

    def train_stacking_classifier(self, X_train, X_test, y_train, y_test,
                                   random_state=42, cv=5, is_tuned=False,
                                   custom_aggregation_name='sum', visualize=True):
        """
        Train Stacking Classifier (combines RF, DT, SVC, KNN, LogReg)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
            random_state: Random seed
            cv: Cross-validation folds for stacking
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Name of aggregation used
            visualize: Whether to generate visualizations

        Returns:
            Trained StackingClassifier model
        """
        print("="*80)
        print(" "*25 + "STACKING CLASSIFIER")
        print("="*80)

        # Define base estimators
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
            ('dt', DecisionTreeClassifier(random_state=random_state)),
            ('svc', svm.SVC(probability=True, random_state=random_state)),
            ('knn', neighbors.KNeighborsClassifier()),
            ('lr', LogisticRegression(max_iter=1000, random_state=random_state))
        ]

        # Define meta-learner
        final_estimator = LogisticRegression(max_iter=1000, random_state=random_state)

        # Train model
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        # Metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        print(f"\nStacking Classifier Performance:")
        print(f"  Base Estimators: RF, DT, SVC, KNN, LogReg")
        print(f"  Meta-Learner: Logistic Regression")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1_score:.4f}")

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_classification_metrics(
            y_test, y_pred, y_score
        )
        self.metrics_collector.add_metrics('Stacking', 'classification', is_tuned, metrics_dict)

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, y_score, 'Stacking', 'classification', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_classification_visualizations(
                model, X_test, y_test, y_pred, y_score,
                'Stacking_Classifier', is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Stacking', 'classification')

        print("="*80 + "\n")
        self.trained_models['Stacking_clf'] = model
        return model

    def train_stacking_regressor(self, X_train, X_test, y_train, y_test,
                                 random_state=42, cv=5, is_tuned=False,
                                 custom_aggregation_name='sum', visualize=True):
        """
        Train Stacking Regressor (combines RF, DT, SVR, KNN, Ridge, Lasso)

        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            random_state: Random seed
            cv: Cross-validation folds for stacking
            is_tuned: Whether this is a tuned model
            custom_aggregation_name: Name of aggregation used
            visualize: Whether to generate visualizations

        Returns:
            Trained StackingRegressor model
        """
        print("="*80)
        print(" "*25 + "STACKING REGRESSOR")
        print("="*80)

        # Define base estimators
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=random_state)),
            ('dt', DecisionTreeRegressor(random_state=random_state)),
            ('svr', svm.SVR()),
            ('knn', neighbors.KNeighborsRegressor()),
            ('ridge', Ridge(random_state=random_state)),
            ('lasso', Lasso(random_state=random_state))
        ]

        # Define meta-learner
        final_estimator = Ridge(random_state=random_state)

        # Train model
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = metrics.r2_score(y_test, y_pred)

        print(f"\nStacking Regressor Performance:")
        print(f"  Base Estimators: RF, DT, SVR, KNN, Ridge, Lasso")
        print(f"  Meta-Learner: Ridge Regression")
        print(f"  MAE:  {mae:.4f}")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Collect metrics
        metrics_dict = self.metrics_collector.collect_regression_metrics(y_test, y_pred)
        self.metrics_collector.add_metrics('Stacking', 'regression', is_tuned, metrics_dict)
        print(f"\n✓ Stacking - RMSE: {metrics_dict['RMSE']:.5f}, "
              f"MAE: {metrics_dict['MAE']:.5f}, R²: {metrics_dict['R2_Score']:.5f}")

        # Save predictions to CSV if enabled in config
        if self.output_config and getattr(self.output_config, 'GENERATE_PREDS', False):
            pred_file = self._save_predictions(y_test, y_pred, None, 'Stacking', 'regression', is_tuned)
            if pred_file:
                print(f"✓ Predictions saved to: {pred_file}")

        # Visualizations
        if visualize and self.output_config:
            self._generate_regression_visualizations(
                y_test, y_pred, 'Stacking_Regression',
                is_tuned, custom_aggregation_name
            )

        # Generate learning curve
        self._generate_learning_curve(model, X_train, y_train, 'Stacking', 'regression')

        print("="*80 + "\n")
        self.trained_models['Stacking_reg'] = model
        return model

    def _generate_classification_visualizations(self, model, X_test, y_test,
                                                y_pred, y_score, model_name,
                                                is_tuned, custom_aggregation_name):
        """Generate all classification visualizations"""
        output_base = self.output_config.CLF_TUNED_DIR if is_tuned else self.output_config.CLF_BASE_DIR

        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred,
            target_classes=sorted(y_test.unique()),
            model_name=model_name,
            output_path=output_base / 'confusion_matrices',
            is_tuned=is_tuned,
            custom_aggregation_name=custom_aggregation_name
        )

        # ROC curve
        plot_roc_curve(
            y_test, y_score,
            model_name=model_name,
            output_path=output_base / 'roc_curves',
            is_tuned=is_tuned,
            custom_aggregation_name=custom_aggregation_name
        )

        # Bar chart
        plot_classification_bar_chart(
            y_test, y_pred,
            model_name=model_name,
            output_path=output_base / 'bar_charts',
            is_tuned=is_tuned,
            custom_aggregation_name=custom_aggregation_name
        )

        # Probability matrix
        plot_probability_matrix(
            model, X_test, y_test,
            model_name=model_name,
            output_path=output_base / 'probability_matrices',
            is_tuned=is_tuned
        )

    def _generate_regression_visualizations(self, y_test, y_pred, model_name,
                                           is_tuned, custom_aggregation_name):
        """Generate all regression visualizations"""
        output_base = self.output_config.REG_TUNED_DIR if is_tuned else self.output_config.REG_BASE_DIR

        # Scatter plot
        plot_regression_scatter(
            y_test, y_pred,
            model_name=model_name,
            output_path=output_base / 'scatter_plots',
            is_tuned=is_tuned,
            custom_aggregation_name=custom_aggregation_name
        )

    def _print_model_parameters(self, model, model_name, is_tuned):
        """Print model parameters"""
        tuning_status = "TUNED" if is_tuned else "DEFAULT"
        print(f"\n{'='*70}")
        print(f"  {model_name} Parameters ({tuning_status})")
        print(f"{'='*70}")

        params = model.get_params()
        important_params = {}

        if 'RandomForest' in model_name or 'DecisionTree' in model_name:
            important_params = {
                'n_estimators': params.get('n_estimators', 'N/A'),
                'max_depth': params.get('max_depth', 'N/A'),
                'min_samples_split': params.get('min_samples_split', 'N/A'),
                'min_samples_leaf': params.get('min_samples_leaf', 'N/A'),
            }
        elif 'SVC' in model_name or 'SVM' in model_name or 'SVR' in str(type(model)):
            important_params = {
                'C': params.get('C', 'N/A'),
                'kernel': params.get('kernel', 'N/A'),
                'gamma': params.get('gamma', 'N/A'),
            }
        elif 'KNN' in model_name:
            important_params = {
                'n_neighbors': params.get('n_neighbors', 'N/A'),
                'weights': params.get('weights', 'N/A'),
                'metric': params.get('metric', 'N/A'),
            }

        for param_name, param_value in important_params.items():
            if param_value != 'N/A':
                print(f"  {param_name:25s}: {param_value}")

        print(f"{'='*70}\n")


