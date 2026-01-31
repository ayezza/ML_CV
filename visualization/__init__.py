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
from .cv_analysis import (
    plot_cv_comparison_bars,
    plot_cv_trend_lines,
    plot_cv_heatmap,
    plot_cv_tuning_time,
    plot_cv_best_params_summary,
    plot_cv_metrics_comparison,
    generate_cv_analysis_report
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
    # CV Analysis
    'plot_cv_comparison_bars',
    'plot_cv_trend_lines',
    'plot_cv_heatmap',
    'plot_cv_tuning_time',
    'plot_cv_best_params_summary',
    'plot_cv_metrics_comparison',
    'generate_cv_analysis_report',
]
