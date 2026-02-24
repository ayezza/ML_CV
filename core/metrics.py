"""
Metrics Collection and Management Module

This module handles collection, storage, and export of model performance metrics.
"""
# Author: Abdel YEZZA
# NOTE: Any use within a professional context must reference the author.

from sklearn import metrics
from sklearn.metrics import (
    precision_recall_fscore_support,
    root_mean_squared_error,
    mean_absolute_error,
    max_error,
    r2_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


class MetricsCollector:
    """
    Collects and manages metrics for ML models

    This class maintains a centralized store of all model metrics and provides
    methods to add new metrics and export them to various formats.
    """

    def __init__(self):
        """Initialize empty metrics store"""
        self.metrics_store = []

    def add_metrics(self, model_name, model_type, is_tuned, metrics_dict, training_time=None):
        """
        Add model metrics to the store

        Args:
            model_name: Name of the model (e.g., 'RandomForest', 'SVC')
            model_type: 'classification' or 'regression'
            is_tuned: True if hyperparameter tuning was applied
            metrics_dict: Dictionary containing the metrics
            training_time: Training time in seconds (optional)

        Example:
            >>> collector = MetricsCollector()
            >>> metrics = {'Accuracy': 0.95, 'F1': 0.93}
            >>> collector.add_metrics('RandomForest', 'classification', False, metrics, 12.5)
        """
        record = {
            'Model': model_name,
            'Type': model_type.capitalize(),
            'Tuned': 'Yes' if is_tuned else 'No',
            'Training_Time_sec': training_time if training_time is not None else 0.0
        }
        record.update(metrics_dict)
        self.metrics_store.append(record)

    def update_last_training_time(self, training_time):
        """
        Update the training time of the most recently added metric

        Args:
            training_time: Training time in seconds

        Example:
            >>> collector.add_metrics('RandomForest', 'classification', False, metrics)
            >>> collector.update_last_training_time(12.5)
        """
        if self.metrics_store:
            self.metrics_store[-1]['Training_Time_sec'] = training_time

    def collect_classification_metrics(self, y_test, y_pred, y_score, n_classes=4):
        """
        Collect all classification metrics

        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_score: Prediction probabilities (for AUC-ROC)
            n_classes: Number of classes (default: 4)

        Returns:
            dict: Dictionary containing all classification metrics
        """

        # Confusion matrix - automatically handles K classes
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)

        # Basic metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision, recall, f1_Macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        precision, recall, f1_Micro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='micro', zero_division=0
        )

        # Per-class metrics
        precision_macro = metrics.precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = metrics.recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Micro-average (equals accuracy for multi-class)
        precision_micro = metrics.precision_score(y_test, y_pred, average='micro', zero_division=0)
        recall_micro = metrics.recall_score(y_test, y_pred, average='micro', zero_division=0)
        f1_micro = metrics.f1_score(y_test, y_pred, average='micro', zero_division=0)

        # Weighted average (by class frequency)
        precision_weighted = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # AUC-ROC (multiclass - multiple averaging strategies)
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

        # Handle binary classification case (label_binarize returns 1D for n_classes=2)
        if n_classes == 2:
            # For binary: use probability of positive class directly
            y_score_for_roc = y_score[:, 1] if y_score.ndim > 1 else y_score
            fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score_for_roc)
        else:
            # For multiclass: ravel both arrays
            fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        auc_roc_micro = auc(fpr_micro, tpr_micro)

        # Macro-average: compute AUC for each class, then average (treats all classes equally)
        auc_roc_macro = 0.0
        try:
            from sklearn.metrics import roc_auc_score
            if n_classes == 2:
                # Binary classification: use probability of positive class
                y_score_binary = y_score[:, 1] if y_score.ndim > 1 else y_score
                auc_roc_macro = roc_auc_score(y_test, y_score_binary)
            else:
                auc_roc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"[WARNING] Could not compute AUC_ROC_Macro: {e}")
            auc_roc_macro = 0.0

        # Weighted-average: compute AUC for each class, then weighted average by class frequency
        auc_roc_weighted = 0.0
        try:
            if n_classes == 2:
                # Binary classification: weighted = macro for binary
                auc_roc_weighted = auc_roc_macro
            else:
                auc_roc_weighted = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"[WARNING] Could not compute AUC_ROC_Weighted: {e}")
            auc_roc_weighted = 0.0

        return {
            'Confusion_Matrix': conf_matrix,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Precision_Macro': precision_macro,
            'Precision_Micro': precision_micro,
            'Precision_Weighted': precision_weighted,
            'Recall_Macro': recall_macro,
            'Recall_Micro': recall_micro,
            'Recall_Weighted': recall_weighted,
            'F1_Macro': f1_macro,
            'F1_Micro': f1_micro,
            'F1_Weighted': f1_weighted,
            'AUC_ROC_Micro': auc_roc_micro,
            'AUC_ROC_Macro': auc_roc_macro,        # Treats all classes equally
            'AUC_ROC_Weighted': auc_roc_weighted   # Weighted by class frequency
        }

    def collect_regression_metrics(self, y_test, y_pred):
        """
        Collect all regression metrics

        Args:
            y_test: True values
            y_pred: Predicted values

        Returns:
            dict: Dictionary containing all regression metrics
        """
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Max_Error': max_err,
            'MAPE': mape,
            'R2_Score': r2
        }

    def get_all_metrics(self):
        """
        Get all collected metrics

        Returns:
            list: List of all metric records
        """
        return self.metrics_store

    def get_classification_metrics(self):
        """
        Get only classification metrics

        Returns:
            list: List of classification metric records
        """
        return [m for m in self.metrics_store if m['Type'] == 'Classification']

    def get_regression_metrics(self):
        """
        Get only regression metrics

        Returns:
            list: List of regression metric records
        """
        return [m for m in self.metrics_store if m['Type'] == 'Regression']

    def get_best_classification_model(self):
        """
        Find the best classification model by accuracy

        Returns:
            dict: Best classification model record, or None if no classification models
        """
        clf_metrics = self.get_classification_metrics()
        if not clf_metrics:
            return None

        return max(clf_metrics, key=lambda x: x.get('Accuracy', 0))

    def get_best_regression_model(self):
        """
        Find the best regression model by R² score

        Returns:
            dict: Best regression model record, or None if no regression models
        """
        reg_metrics = self.get_regression_metrics()
        if not reg_metrics:
            return None

        return max(reg_metrics, key=lambda x: x.get('R2_Score', 0))

    def print_summary(self):
        """Print a summary of collected metrics"""
        print("\n" + "="*80)
        print(" "*25 + "METRICS SUMMARY")
        print("="*80)

        clf_metrics = self.get_classification_metrics()
        reg_metrics = self.get_regression_metrics()

        print(f"Total models evaluated: {len(self.metrics_store)}")
        print(f"  - Classification models: {len(clf_metrics)}")
        print(f"  - Regression models: {len(reg_metrics)}")

        if clf_metrics:
            best_clf = self.get_best_classification_model()
            print(f"\n✓ Best Classification Model: {best_clf['Model']} "
                  f"(Tuned: {best_clf['Tuned']}) - Accuracy: {best_clf['Accuracy']:.5f}")

        if reg_metrics:
            best_reg = self.get_best_regression_model()
            print(f"✓ Best Regression Model: {best_reg['Model']} "
                  f"(Tuned: {best_reg['Tuned']}) - R²: {best_reg['R2_Score']:.5f}")

        print("="*80 + "\n")

    def clear(self):
        """Clear all collected metrics"""
        self.metrics_store = []
