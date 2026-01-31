"""
Cross-Validation Analysis Visualization Module

This module provides visualization functions for CV experiment results,
comparing different CV fold values across multiple models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_cv_comparison_bars(results_df, output_path, metric='Test_F1',
                            title='CV Fold Comparison by Model',
                            figsize=(12, 7)):
    """
    Create grouped bar chart comparing CV values across models.

    Args:
        results_df: DataFrame with columns ['Model', 'CV_Folds', metric]
        output_path: Directory path where plot will be saved
        metric: Column name for the metric to plot (default: 'Test_F1')
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot

    Example:
        >>> plot_cv_comparison_bars(df, './output/graphs', metric='Test_F1')
    """
    print("="*80)
    print(" "*20 + "CV COMPARISON BAR CHART")
    print("="*80)

    # Pivot data for grouped bar chart
    pivot_df = results_df.pivot(index='CV_Folds', columns='Model', values=metric)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get positions and width
    x = np.arange(len(pivot_df.index))
    n_models = len(pivot_df.columns)
    width = 0.8 / n_models

    # Color palette
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Plot bars for each model
    for i, (model, color) in enumerate(zip(pivot_df.columns, colors)):
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot_df[model], width, label=model, color=color, edgecolor='black', linewidth=0.5)

        # Add value labels on bars using pivoted data and model name
        for bar, val in zip(bars, pivot_df[model]):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, rotation=0)

    # Customize plot
    ax.set_xlabel('CV Folds', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index.astype(int))
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-axis to start near minimum value for better visualization
    y_min = pivot_df.min().min()
    y_max = pivot_df.max().max()
    margin = (y_max - y_min) * 0.15
    ax.set_ylim(max(0, y_min - margin), y_max + margin)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f'cv_comparison_bars_{metric.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] CV comparison bar chart saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_cv_trend_lines(results_df, output_path, metric='Test_F1',
                        title='CV Fold Trend Analysis',
                        figsize=(12, 7)):
    """
    Create line plot showing metric trends across CV values for each model.

    Args:
        results_df: DataFrame with columns ['Model', 'CV_Folds', metric]
        output_path: Directory path where plot will be saved
        metric: Column name for the metric to plot (default: 'Test_F1')
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot
    """
    print("="*80)
    print(" "*20 + "CV TREND LINE PLOT")
    print("="*80)

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique models
    models = results_df['Model'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']

    for i, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model].sort_values('CV_Folds')
        ax.plot(model_data['CV_Folds'], model_data[metric],
                marker=markers[i % len(markers)], markersize=10,
                linewidth=2.5, label=model, color=colors[i])

        # Add value annotations
        for x, y in zip(model_data['CV_Folds'], model_data[metric]):
            ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.8)

    ax.set_xlabel('CV Folds', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' '), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')

    # Set x-ticks to actual CV values
    cv_values = sorted(results_df['CV_Folds'].unique())
    ax.set_xticks(cv_values)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f'cv_trend_lines_{metric.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] CV trend line plot saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_cv_heatmap(results_df, output_path, metric='Test_F1',
                    title='Model Performance Heatmap (CV vs Model)',
                    figsize=(10, 8)):
    """
    Create heatmap showing metric values for each model/CV combination.

    Args:
        results_df: DataFrame with columns ['Model', 'CV_Folds', metric]
        output_path: Directory path where plot will be saved
        metric: Column name for the metric to plot (default: 'Test_F1')
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot
    """
    print("="*80)
    print(" "*20 + "CV PERFORMANCE HEATMAP")
    print("="*80)

    # Pivot data for heatmap
    pivot_df = results_df.pivot(index='Model', columns='CV_Folds', values=metric)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, cbar_kws={'label': metric.replace('_', ' ')},
                annot_kws={'size': 10, 'weight': 'bold'})

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('CV Folds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f'cv_heatmap_{metric.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] CV heatmap saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_cv_tuning_time(results_df, output_path,
                        title='Tuning Time by CV Folds',
                        figsize=(12, 7)):
    """
    Create bar chart showing tuning time for each model/CV combination.

    Args:
        results_df: DataFrame with columns ['Model', 'CV_Folds', 'Tuning_Time_sec']
        output_path: Directory path where plot will be saved
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot
    """
    print("="*80)
    print(" "*20 + "CV TUNING TIME COMPARISON")
    print("="*80)

    # Pivot data
    pivot_df = results_df.pivot(index='CV_Folds', columns='Model', values='Tuning_Time_sec')

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked bar or grouped bar
    pivot_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('CV Folds', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tuning Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(title='Models', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Rotate x-axis labels
    plt.xticks(rotation=0)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / 'cv_tuning_time.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] CV tuning time chart saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_cv_best_params_summary(results_df, output_path,
                                title='Best Parameters Summary',
                                figsize=(14, 8)):
    """
    Create a table visualization of best parameters for each model/CV combination.

    Args:
        results_df: DataFrame with columns ['Model', 'CV_Folds', 'Best_Params', 'Test_F1']
        output_path: Directory path where plot will be saved
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot
    """
    print("="*80)
    print(" "*20 + "BEST PARAMETERS SUMMARY")
    print("="*80)

    # Create summary table
    summary = results_df.pivot_table(
        index='Model',
        columns='CV_Folds',
        values='Test_F1',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Find best CV for each model
    best_cv_per_model = summary.idxmax(axis=1)
    best_score_per_model = summary.max(axis=1)

    # Create table data
    table_data = []
    for model in summary.index:
        row = [model, int(best_cv_per_model[model]), f'{best_score_per_model[model]:.4f}']
        # Get best params for best CV
        best_params = results_df[
            (results_df['Model'] == model) &
            (results_df['CV_Folds'] == best_cv_per_model[model])
        ]['Best_Params'].values[0]
        row.append(str(best_params)[:60] + '...' if len(str(best_params)) > 60 else str(best_params))
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Best CV', 'Best F1', 'Best Parameters'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.1, 0.12, 0.63]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / 'cv_best_params_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Best parameters summary saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_cv_metrics_comparison(results_df, output_path,
                               metrics=['Test_Accuracy', 'Test_F1', 'Test_AUC'],
                               title='Multi-Metric CV Comparison',
                               figsize=(14, 10)):
    """
    Create subplot grid comparing multiple metrics across CV values.

    Args:
        results_df: DataFrame with metric columns
        output_path: Directory path where plot will be saved
        metrics: List of metric column names to compare
        title: Title for the plot
        figsize: Figure size tuple

    Returns:
        Path: Full path to the saved plot
    """
    print("="*80)
    print(" "*20 + "MULTI-METRIC CV COMPARISON")
    print("="*80)

    # Filter metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in results_df.columns]

    if not available_metrics:
        print("No valid metrics found in results DataFrame")
        return None

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    models = results_df['Model'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, available_metrics):
        for i, model in enumerate(models):
            model_data = results_df[results_df['Model'] == model].sort_values('CV_Folds')
            ax.plot(model_data['CV_Folds'], model_data[metric],
                   marker='o', markersize=8, linewidth=2,
                   label=model, color=colors[i])

        ax.set_xlabel('CV Folds', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_title(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)

        cv_values = sorted(results_df['CV_Folds'].unique())
        ax.set_xticks(cv_values)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / 'cv_metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Multi-metric comparison saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def generate_cv_analysis_report(results_df, output_path, task_type='classification'):
    """
    Generate all CV analysis visualizations.

    Args:
        results_df: DataFrame with CV experiment results
        output_path: Directory path where plots will be saved
        task_type: 'classification' or 'regression'

    Returns:
        dict: Dictionary of saved plot paths
    """
    print("\n" + "="*80)
    print(" "*15 + "GENERATING CV ANALYSIS REPORT")
    print("="*80 + "\n")

    saved_plots = {}

    # Filter by task type
    task_results = results_df[results_df['Task'] == task_type.capitalize()] if 'Task' in results_df.columns else results_df

    if task_results.empty:
        print(f"No results found for task type: {task_type}")
        return saved_plots

    # Determine metric based on task type
    if task_type == 'classification':
        primary_metric = 'Test_F1' # prefer F1 for classification that balances precision and recall
        metrics = ['Test_Accuracy', 'Test_F1', 'Test_AUC']
    else:
        primary_metric = 'Test_R2' if 'Test_R2' in task_results.columns else 'Best_CV_Score'
        metrics = ['Test_R2', 'Test_RMSE', 'Test_MAE'] if 'Test_R2' in task_results.columns else ['Best_CV_Score']

    # 1. Bar chart comparison
    try:
        saved_plots['bars'] = plot_cv_comparison_bars(
            task_results, output_path, metric=primary_metric,
            title=f'CV Comparison - {primary_metric.replace("_", " ")}'
        )
    except Exception as e:
        print(f"[WARNING] Could not generate bar chart: {e}")

    # 2. Trend lines
    try:
        saved_plots['trends'] = plot_cv_trend_lines(
            task_results, output_path, metric=primary_metric,
            title=f'CV Trend Analysis - {primary_metric.replace("_", " ")}'
        )
    except Exception as e:
        print(f"[WARNING] Could not generate trend lines: {e}")

    # 3. Heatmap
    try:
        saved_plots['heatmap'] = plot_cv_heatmap(
            task_results, output_path, metric=primary_metric,
            title=f'Performance Heatmap - {primary_metric.replace("_", " ")}'
        )
    except Exception as e:
        print(f"[WARNING] Could not generate heatmap: {e}")

    # 4. Tuning time
    if 'Tuning_Time_sec' in task_results.columns:
        try:
            saved_plots['tuning_time'] = plot_cv_tuning_time(
                task_results, output_path,
                title='Model Tuning Time by CV Folds'
            )
        except Exception as e:
            print(f"[WARNING] Could not generate tuning time chart: {e}")

    # 5. Best parameters summary
    if 'Best_Params' in task_results.columns:
        try:
            saved_plots['params_summary'] = plot_cv_best_params_summary(
                task_results, output_path,
                title='Best Parameters Summary per Model'
            )
        except Exception as e:
            print(f"[WARNING] Could not generate params summary: {e}")

    # 6. Multi-metric comparison
    available_metrics = [m for m in metrics if m in task_results.columns]
    if len(available_metrics) > 1:
        try:
            saved_plots['multi_metric'] = plot_cv_metrics_comparison(
                task_results, output_path, metrics=available_metrics,
                title=f'Multi-Metric Comparison ({task_type.capitalize()})'
            )
        except Exception as e:
            print(f"[WARNING] Could not generate multi-metric comparison: {e}")

    print("\n" + "="*80)
    print(f" Generated {len(saved_plots)} CV analysis plots")
    print("="*80 + "\n")

    return saved_plots
