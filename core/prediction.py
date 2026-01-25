"""
Prediction Module

This module handles making predictions with trained models and managing best models.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class ModelPredictor:
    """
    Handles predictions using trained models

    This class provides methods to save best models, load them, and make predictions
    on new data with proper preprocessing.
    """

    def __init__(self, output_path=None):
        """
        Initialize the predictor

        Args:
            output_path: Path where models will be saved (optional)
        """
        self.output_path = Path(output_path) if output_path else Path('./output/models')
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.best_classification_model = None
        self.best_regression_model = None
        self.feature_names = None
        self.scaler = None

    def save_model(self, model, model_name, model_type='classification'):
        """
        Save a trained model to disk

        Args:
            model: Trained model object
            model_name: Name for the saved model file
            model_type: 'classification' or 'regression'

        Returns:
            Path: Full path to the saved model file

        Example:
            >>> predictor = ModelPredictor()
            >>> predictor.save_model(best_rf_clf, 'RandomForest_best', 'classification')
        """
        filename = f"{model_name}_{model_type}.pkl"
        model_path = self.output_path / filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"✓ Model saved to: {model_path}")
        return model_path

    def load_model(self, model_path):
        """
        Load a trained model from disk

        Args:
            model_path: Path to the saved model file

        Returns:
            Loaded model object

        Example:
            >>> predictor = ModelPredictor()
            >>> model = predictor.load_model('./output/models/RandomForest_best_classification.pkl')
        """
        model_path = Path(model_path)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"✓ Model loaded from: {model_path}")
        return model

    def set_best_models(self, metrics_collector):
        """
        Identify and save the best classification and regression models

        Args:
            metrics_collector: MetricsCollector instance with all model metrics

        Returns:
            dict: Dictionary with best model information

        Example:
            >>> predictor = ModelPredictor()
            >>> best_models = predictor.set_best_models(metrics_collector)
        """
        print("="*80)
        print(" "*25 + "IDENTIFYING BEST MODELS")
        print("="*80)

        best_models = {}

        # Get best classification model
        best_clf = metrics_collector.get_best_classification_model()
        if best_clf:
            best_models['classification'] = {
                'model_name': best_clf['Model'],
                'is_tuned': best_clf['Tuned'] == 'Yes',
                'accuracy': best_clf['Accuracy'],
                'metrics': best_clf
            }
            print(f"\n✓ Best Classification Model: {best_clf['Model']}")
            print(f"  Tuned: {best_clf['Tuned']}")
            print(f"  Accuracy: {best_clf['Accuracy']:.5f}")

        # Get best regression model
        best_reg = metrics_collector.get_best_regression_model()
        if best_reg:
            best_models['regression'] = {
                'model_name': best_reg['Model'],
                'is_tuned': best_reg['Tuned'] == 'Yes',
                'r2_score': best_reg['R2_Score'],
                'metrics': best_reg
            }
            print(f"\n✓ Best Regression Model: {best_reg['Model']}")
            print(f"  Tuned: {best_reg['Tuned']}")
            print(f"  R² Score: {best_reg['R2_Score']:.5f}")

        print("="*80 + "\n")
        return best_models

    def predict_with_model(self, model, new_data, model_type='classification',
                          feature_names=None, return_proba=True):
        """
        Make predictions on new data using a trained model

        Args:
            model: Trained model
            new_data: New data for prediction (DataFrame, dict, or numpy array)
            model_type: 'classification' or 'regression'
            feature_names: List of expected feature names (for validation)
            return_proba: Whether to return probabilities for classification

        Returns:
            dict: Dictionary with predictions and probabilities (if applicable)

        Example:
            >>> predictor = ModelPredictor()
            >>> new_data = pd.DataFrame({'X1': [0.5], 'X2': [0.6], ...})
            >>> result = predictor.predict_with_model(model, new_data, 'classification')
        """
        print("="*80)
        print(" "*25 + "MAKING PREDICTIONS")
        print("="*80)

        # Convert input to DataFrame if needed
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        elif isinstance(new_data, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names required when input is numpy array")
            new_data = pd.DataFrame(new_data, columns=feature_names)

        # Validate features
        if feature_names:
            missing_features = set(feature_names) - set(new_data.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

        print(f"\nInput data shape: {new_data.shape}")
        print(f"Model type: {model_type}")

        # Make predictions
        predictions = model.predict(new_data)

        result = {
            'predictions': predictions,
            'input_data': new_data
        }

        if model_type == 'classification':
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(new_data)
                result['probabilities'] = probabilities

                # Get class labels if available
                if hasattr(model, 'classes_'):
                    result['classes'] = model.classes_

                print(f"\nPredictions: {predictions}")
                print(f"Probabilities shape: {probabilities.shape}")

                # Display detailed predictions
                print("\n" + "-"*80)
                print("DETAILED PREDICTIONS:")
                print("-"*80)
                for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                    print(f"\nSample {i+1}:")
                    print(f"  Predicted class: {pred}")
                    if hasattr(model, 'classes_'):
                        for cls, prob in zip(model.classes_, proba):
                            print(f"  Class {cls} probability: {prob:.4f}")
            else:
                print(f"\nPredictions: {predictions}")
        else:  # regression
            print(f"\nPredictions: {predictions}")
            print(f"  Min: {predictions.min():.4f}")
            print(f"  Max: {predictions.max():.4f}")
            print(f"  Mean: {predictions.mean():.4f}")

        print("="*80 + "\n")
        return result

    def save_predictions(self, predictions_result, output_path, filename='predictions.csv'):
        """
        Save predictions to CSV file

        Args:
            predictions_result: Result dictionary from predict_with_model()
            output_path: Directory where to save the file
            filename: Name of the CSV file

        Returns:
            Path: Full path to the saved file

        Example:
            >>> predictor.save_predictions(result, Path('./output/predictions'), 'new_predictions.csv')
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create DataFrame with predictions
        df = predictions_result['input_data'].copy()
        df['Prediction'] = predictions_result['predictions']

        # Add probabilities if available (depending on model type)
        if 'probabilities' in predictions_result:
            proba = predictions_result['probabilities']
            classes = predictions_result.get('classes', range(proba.shape[1]))
            for i, cls in enumerate(classes):
                df[f'Probability_Class_{cls}'] = proba[:, i]

        # Save to CSV
        save_path = output_path / filename
        df.to_csv(save_path, index=False)

        print(f"✓ Predictions saved to: {save_path}")
        return save_path

    def create_prediction_report(self, predictions_result, model_name, output_path,
                                filename='prediction_report.txt'):
        """
        Create a detailed prediction report

        Args:
            predictions_result: Result dictionary from predict_with_model()
            model_name: Name of the model used
            output_path: Directory where to save the report
            filename: Name of the report file

        Returns:
            Path: Full path to the saved report

        Example:
            >>> predictor.create_prediction_report(result, 'RandomForest', Path('./output/reports'))
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        save_path = output_path / filename

        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "PREDICTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Number of predictions: {len(predictions_result['predictions'])}\n\n")

            f.write("-"*80 + "\n")
            f.write("PREDICTIONS:\n")
            f.write("-"*80 + "\n\n")

            predictions = predictions_result['predictions']
            probabilities = predictions_result.get('probabilities')
            classes = predictions_result.get('classes')

            for i, pred in enumerate(predictions):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Predicted value/class: {pred}\n")

                if probabilities is not None:
                    f.write(f"  Probabilities:\n")
                    if classes is not None:
                        for cls, prob in zip(classes, probabilities[i]):
                            f.write(f"    Class {cls}: {prob:.4f}\n")
                    else:
                        for j, prob in enumerate(probabilities[i]):
                            f.write(f"    Class {j}: {prob:.4f}\n")
                f.write("\n")

            f.write("="*80 + "\n")

        print(f"✓ Prediction report saved to: {save_path}")
        return save_path


# Convenience function for quick predictions
def predict_new_data(model, new_data, model_type='classification',
                    feature_names=None, save_path=None):
    """
    Quick function to make predictions on new data

    Args:
        model: Trained model
        new_data: New data (DataFrame, dict, or array)
        model_type: 'classification' or 'regression'
        feature_names: List of feature names (if new_data is array)
        save_path: Path to save predictions (optional)

    Returns:
        Prediction results dictionary

    Example:
        >>> from core.prediction import predict_new_data
        >>> new_sample = {'X1': 0.98, 'X2': 514.5, ...}
        >>> result = predict_new_data(best_model, new_sample, 'classification')
    """
    predictor = ModelPredictor()
    result = predictor.predict_with_model(
        model, new_data, model_type, feature_names, return_proba=True
    )

    if save_path:
        predictor.save_predictions(result, save_path)

    return result
