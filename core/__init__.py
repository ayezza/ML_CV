"""
Core ML functionality package
"""
from .metrics import (
    MetricsCollector
)
from .preprocessing import (
    DataPreprocessor
)
from .models import (
    ModelTrainer
)
from .tuning import (
    ModelTuner
)
from .prediction import (
    ModelPredictor,
    predict_new_data
)
from .multioutput import (
    MultiOutputPredictor,
    create_multioutput_targets
)

__version__ = '1.0.0'

__all__ = [
    # Metrics
    'MetricsCollector',
    'add_model_metrics',
    'collect_classification_metrics',
    'collect_regression_metrics',
    'get_global_collector',
    # Preprocessing
    'DataPreprocessor',
    'load_data',
    'compute_correlations',
    'create_target_variables',
    'preprocessing_step',
    # Models
    'ModelTrainer',
    'modeling_step',
    # Tuning
    'ModelTuner',
    'tuned_modeling_step',
    # Prediction
    'ModelPredictor',
    'predict_new_data',
    # Multi-output
    'MultiOutputPredictor',
    'create_multioutput_targets',
]
