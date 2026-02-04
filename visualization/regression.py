"""
Regression Visualization Module

This module provides visualization functions for regression models.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from pathlib import Path


def plot_regression_scatter(y_test, y_pred, model_name, output_path,
                           is_tuned=False, custom_aggregation_name='sum',
                           title=None, show_residuals=False, dataset_info=None):
    """
    Generate scatter plot for regression showing actual vs predicted values

    Args:
        y_test: True target values
        y_pred: Predicted target values
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        custom_aggregation_name: Name of aggregation function used
        title: Custom title (optional)
        show_residuals: Whether to add residual plot (default: False)
        dataset_info: Dict with 'name', 'target_cols', 'filename_suffix' (optional)

    Returns:
        Path: Full path to the saved scatter plot

    Example:
        >>> plot_regression_scatter(
        ...     y_test, y_pred,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/regression/base/scatter_plots'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*25 + "REGRESSION SCATTER PLOT")
    print("="*80)

    if show_residuals:
        # Create subplot with residuals
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Main scatter plot
        ax = ax1
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=80)

        # Add perfect prediction line (diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Add metrics as text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=props)

        ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')

        if title is None:
            title = f'Actual vs Predicted - {model_name}'

        # Build full title with dataset info if provided
        full_title = f'{title}\n(Aggregation: {custom_aggregation_name})'
        if dataset_info:
            full_title = f"{title}\nDataset: {dataset_info['name']} | Target: {dataset_info['target_cols']}\n(Aggregation: {custom_aggregation_name})"
        ax.set_title(full_title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Residual plot
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5, s=80)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add residual statistics
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        textstr_res = f'Mean = {residual_mean:.4f}\nStd = {residual_std:.4f}'
        props_res = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax2.text(0.05, 0.95, textstr_res, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top', bbox=props_res)

        plt.tight_layout()
    else:
        # Single scatter plot
        plt.figure(figsize=(10, 8))

        # Create scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=80)

        # Add perfect prediction line (diagonal)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='Perfect Prediction')

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Add metrics as text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=props)

        plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')

        if title is None:
            title = f'Actual vs Predicted - {model_name}'

        # Build full title with dataset info if provided
        full_title = f'{title}\n(Aggregation: {custom_aggregation_name})'
        if dataset_info:
            full_title = f"{title}\nDataset: {dataset_info['name']} | Target: {dataset_info['target_cols']}\n(Aggregation: {custom_aggregation_name})"
        plt.title(full_title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure with dataset info in filename if provided
    suffix = f"_{dataset_info['filename_suffix']}" if dataset_info else ""
    filename = f'regression_scatter_{model_name}_{custom_aggregation_name}{suffix}.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Regression scatter plot saved to: {save_path}")
    print(f"  R² Score: {r2:.5f}")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  MAE: {mae:.5f}")
    print("="*80 + "\n")

    return save_path


def plot_residuals(y_test, y_pred, model_name, output_path,
                  is_tuned=False, custom_aggregation_name='sum'):
    """
    Generate residual plot for regression model

    Args:
        y_test: True target values
        y_pred: Predicted target values
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        custom_aggregation_name: Name of aggregation function used

    Returns:
        Path: Full path to the saved residual plot

    Example:
        >>> plot_residuals(
        ...     y_test, y_pred,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/regression/base/scatter_plots'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*25 + "RESIDUAL PLOT")
    print("="*80)

    # Calculate residuals
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 7))

    # Scatter plot of residuals
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5, s=80)

    # Add horizontal line at 0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')

    # Add statistics as text box
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    textstr = f'Mean = {residual_mean:.4f}\nStd Dev = {residual_std:.4f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xlabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    plt.title(f'Residual Plot - {model_name}\n(Aggregation: {custom_aggregation_name})',
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f'residual_plot_{model_name}_{custom_aggregation_name}.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Residual plot saved to: {save_path}")
    print(f"  Residual Mean: {residual_mean:.5f}")
    print(f"  Residual Std Dev: {residual_std:.5f}")
    print("="*80 + "\n")

    return save_path


def plot_prediction_error(y_test, y_pred, model_name, output_path,
                         is_tuned=False, custom_aggregation_name='sum'):
    """
    Generate prediction error distribution histogram

    Args:
        y_test: True target values
        y_pred: Predicted target values
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        custom_aggregation_name: Name of aggregation function used

    Returns:
        Path: Full path to the saved histogram

    Example:
        >>> plot_prediction_error(
        ...     y_test, y_pred,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/regression/base/scatter_plots'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*20 + "PREDICTION ERROR DISTRIBUTION")
    print("="*80)

    # Calculate prediction errors
    errors = y_test - y_pred

    plt.figure(figsize=(10, 7))

    # Create histogram
    n, bins, patches = plt.hist(errors, bins=30, edgecolor='black',
                                alpha=0.7, color='steelblue')

    # Add vertical line at 0
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')

    # Calculate statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mae = mean_absolute_error(y_test, y_pred)

    # Add statistics as text box
    textstr = f'Mean Error = {mean_error:.4f}\nStd Dev = {std_error:.4f}\nMAE = {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.70, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

    plt.xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Prediction Error Distribution - {model_name}\n(Aggregation: {custom_aggregation_name})',
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f'error_distribution_{model_name}_{custom_aggregation_name}.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Error distribution plot saved to: {save_path}")
    print(f"  Mean Error: {mean_error:.5f}")
    print(f"  Error Std Dev: {std_error:.5f}")
    print("="*80 + "\n")

    return save_path


# Legacy function names for backward compatibility
def generate_regression_scatter_plot(y_test, y_pred, model_name,
                                     title="Actual vs Predicted",
                                     custom_aggregation_name='sum',
                                     output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    return plot_regression_scatter(y_test, y_pred, model_name, output_path,
                                   custom_aggregation_name=custom_aggregation_name,
                                   title=title)
