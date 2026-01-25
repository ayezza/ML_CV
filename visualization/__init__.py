"""
Visualization package for ML models and data analysis
"""
from .classification import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_classification_bar_chart,
    plot_probability_matrix,
    # Legacy names
    generate_conf_matrix,
    generate_auc_roc_curve_multiclass,
    generate_classification_bar_chart,
    generate_probability_matrix
)
from .regression import (
    plot_regression_scatter,
    plot_residuals,
    plot_prediction_error,
    # Legacy names
    generate_regression_scatter_plot
)
from .analysis import (
    plot_correlation_heatmap,
    plot_target_distribution,
    plot_feature_distributions,
    plot_pairplot,
    plot_boxplots,
    # Legacy names
    target_variable_analysis
)

__version__ = '1.0.0'

__all__ = [
    # Classification
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_classification_bar_chart',
    'plot_probability_matrix',
    'generate_conf_matrix',
    'generate_auc_roc_curve_multiclass',
    'generate_classification_bar_chart',
    'generate_probability_matrix',
    # Regression
    'plot_regression_scatter',
    'plot_residuals',
    'plot_prediction_error',
    'generate_regression_scatter_plot',
    # Analysis
    'plot_correlation_heatmap',
    'plot_target_distribution',
    'plot_feature_distributions',
    'plot_pairplot',
    'plot_boxplots',
    'target_variable_analysis',
]
