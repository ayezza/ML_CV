"""
Data Analysis Visualization Module

This module provides visualization functions for data analysis and exploration.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import learning_curve


def plot_correlation_heatmap(df_corr, output_path, filename='correlation_heatmap.png',
                             title='Correlation Heatmap', figsize=(12, 10)):
    """
    Generate correlation heatmap for DataFrame

    Args:
        df_corr: DataFrame containing correlation matrix
        output_path: Directory path where plot will be saved
        filename: Name of the output file (default: 'correlation_heatmap.png')
        title: Title for the plot (default: 'Correlation Heatmap')
        figsize: Figure size tuple (default: (12, 10))

    Returns:
        Path: Full path to the saved heatmap

    Example:
        >>> df_corr = df.corr()
        >>> plot_correlation_heatmap(
        ...     df_corr,
        ...     output_path=Path('./output/graphs'),
        ...     title='Feature Correlations'
        ... )
    """
    print("="*80)
    print(" "*25 + "CORRELATION HEATMAP")
    print("="*80)

    plt.figure(figsize=figsize)
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1, center=0)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Correlation heatmap saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_target_distribution(df, target_column, output_path,
                            custom_aggregation_name='sum',
                            bins=30, kde=True, figsize=(10, 7)):
    """
    Analyze and visualize target variable distribution

    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
        output_path: Directory path where plot will be saved
        custom_aggregation_name: Name of aggregation function used
        bins: Number of bins for histogram (default: 30)
        kde: Whether to show KDE curve (default: True)
        figsize: Figure size tuple (default: (10, 7))

    Returns:
        Path: Full path to the saved plot

    Example:
        >>> plot_target_distribution(
        ...     df,
        ...     target_column='charges_classes',
        ...     output_path=Path('./output/graphs'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*20 + f"TARGET DISTRIBUTION: {target_column}")
    print("="*80)

    target_series = df[target_column]

    # Display statistics
    print("\nTarget variable value counts:")
    print(target_series.value_counts().sort_index())
    print("\nTarget variable description:")
    print(target_series.describe())

    # Create visualization
    plt.figure(figsize=figsize)
    sns.histplot(target_series, bins=bins, kde=kde, color='steelblue',
                edgecolor='black', alpha=0.7)

    plt.title(f"Distribution of Target Variable: {target_column}\n(Aggregation: {custom_aggregation_name})",
             fontsize=14, fontweight='bold')
    plt.xlabel(target_column, fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f"{target_column}_{custom_aggregation_name}_distribution.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Target distribution plot saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_feature_distributions(df, output_path, columns=None,
                               figsize=(16, 12), ncols=3):
    """
    Plot distribution of multiple features in a grid

    Args:
        df: DataFrame containing the data
        output_path: Directory path where plot will be saved
        columns: List of column names to plot (default: all numeric columns)
        figsize: Figure size tuple (default: (16, 12))
        ncols: Number of columns in the grid (default: 3)

    Returns:
        Path: Full path to the saved plot

    Example:
        >>> plot_feature_distributions(
        ...     df,
        ...     output_path=Path('./output/graphs'),
        ...     columns=['X1', 'X2', 'X3', 'X4']
        ... )
    """
    print("="*80)
    print(" "*20 + "FEATURE DISTRIBUTIONS")
    print("="*80)

    if columns is None:
        # Get all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color='steelblue',
                    edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = 'feature_distributions.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Feature distributions plot saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_pairplot(df, output_path, columns=None, hue=None,
                 filename='pairplot.png', sample_size=None):
    """
    Create pairwise scatter plots for features

    Args:
        df: DataFrame containing the data
        output_path: Directory path where plot will be saved
        columns: List of column names to include (default: all numeric)
        hue: Column name for color coding (optional)
        filename: Name of the output file (default: 'pairplot.png')
        sample_size: Number of samples to use (default: all data)

    Returns:
        Path: Full path to the saved plot

    Example:
        >>> plot_pairplot(
        ...     df,
        ...     output_path=Path('./output/graphs'),
        ...     columns=['X1', 'X2', 'X3'],
        ...     hue='charges_classes'
        ... )
    """
    print("="*80)
    print(" "*25 + "PAIRPLOT")
    print("="*80)

    # Sample data if needed
    df_plot = df.sample(n=sample_size, random_state=42) if sample_size else df

    if columns is None:
        # Get all numeric columns
        columns = df_plot.select_dtypes(include=[np.number]).columns.tolist()

    # Create pairplot
    if hue and hue in df_plot.columns:
        plot_df = df_plot[columns + [hue]]
        g = sns.pairplot(plot_df, hue=hue, diag_kind='kde', corner=True,
                        plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k', 'linewidth': 0.5})
    else:
        plot_df = df_plot[columns]
        g = sns.pairplot(plot_df, diag_kind='kde', corner=True,
                        plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k', 'linewidth': 0.5})

    g.fig.suptitle('Pairwise Feature Relationships', fontsize=16, fontweight='bold', y=1.01)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Pairplot saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_boxplots(df, output_path, columns=None, figsize=(16, 10), ncols=3):
    """
    Create boxplots for multiple features to detect outliers

    Args:
        df: DataFrame containing the data
        output_path: Directory path where plot will be saved
        columns: List of column names to plot (default: all numeric columns)
        figsize: Figure size tuple (default: (16, 10))
        ncols: Number of columns in the grid (default: 3)

    Returns:
        Path: Full path to the saved plot

    Example:
        >>> plot_boxplots(
        ...     df,
        ...     output_path=Path('./output/graphs'),
        ...     columns=['X1', 'X2', 'X3', 'X4']
        ... )
    """
    print("="*80)
    print(" "*25 + "BOXPLOTS")
    print("="*80)

    if columns is None:
        # Get all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(columns)
    nrows = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        sns.boxplot(y=df[col], ax=ax, color='lightblue')
        ax.set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Boxplots - Outlier Detection', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = 'feature_boxplots.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[SAVED] Boxplots saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


# Legacy function names for backward compatibility
def target_variable_analysis(df, target_column='charges_classes',
                            custom_aggregation_name='sum', output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    return plot_target_distribution(df, target_column, output_path,
                                    custom_aggregation_name)


def plot_learning_curve(estimator, X, y, model_name, model_type='classification',
                        output_path=None, cv=5, n_jobs=-1,
                        train_sizes=None, scoring=None, figsize=(10, 6), search_type='grid'):
    """
    Generate learning curve plot showing training and cross-validation scores
    vs training set size.

    This helps diagnose:
    - Overfitting (large gap between training and CV scores)
    - Underfitting (both scores are low)
    - Whether more training data would help

    Args:
        estimator: Trained sklearn model or pipeline
        X: Feature matrix (training data)
        y: Target variable (training labels)
        model_name: Name of the model (e.g., 'RandomForest', 'SVC')
        model_type: Type of model - 'classification' or 'regression'
        output_path: Directory where plot will be saved (default: None)
        cv: Number of cross-validation folds (default: 5)
        n_jobs: Number of parallel jobs (default: -1, use all cores)
        train_sizes: Array of training set sizes to evaluate
                    (default: np.linspace(0.1, 1.0, 10))
        scoring: Scoring metric (default: None, uses estimator's score method)
                For classification: 'accuracy', 'f1_weighted', 'roc_auc'
                For regression: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
        figsize: Figure size tuple (default: (10, 6))
        search_type: Hyperparameter search method - 'grid' or 'random' (default: 'grid')

    Returns:
        Path: Full path to the saved plot, or None if output_path not specified

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> model.fit(X_train, y_train)
        >>> plot_learning_curve(
        ...     model, X_train, y_train,
        ...     model_name='RandomForest',
        ...     model_type='classification',
        ...     output_path=Path('./output/graphs/learning_curves'),
        ...     cv=5
        ... )
    """
    print("="*80)
    print(f" LEARNING CURVE: {model_name} ({model_type})".center(80))
    print("="*80)

    # Default train sizes: 10%, 20%, ..., 100% of training data
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    # Default scoring metrics
    if scoring is None:
        if model_type == 'classification':
            scoring = 'accuracy'
        else:  # regression
            scoring = 'r2'

    # Compute learning curve
    print(f"Computing learning curve with {cv}-fold CV...")
    print(f"Train sizes: {train_sizes}")
    print(f"Scoring metric: {scoring}")

    try:
        # Note: For classification with stratified CV, sklearn may limit max train_size
        # to ensure each validation fold has enough samples per class.
        # With cv=3 and n_classes=4, max usable train_size is typically ~67%
        # This is expected behavior and sufficient for diagnostic purposes.
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y,
            train_sizes=train_sizes,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            shuffle=True,
            random_state=42,
            verbose=0
        )
    except Exception as e:
        print(f"[ERROR] Failed to compute learning curve: {e}")
        return None

    # Calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Print statistics
    print(f"\nLearning Curve Statistics:")
    print(f"  Total training set size: {len(X)}")
    print(f"  Requested train sizes: {train_sizes}")
    print(f"  Actual train sizes (absolute): {train_sizes_abs}")
    print(f"  Actual train sizes (percent): {[(s/len(X))*100 for s in train_sizes_abs]}")
    print(f"  Min training samples: {train_sizes_abs[0]} ({(train_sizes_abs[0]/len(X))*100:.1f}%)")
    print(f"  Max training samples: {train_sizes_abs[-1]} ({(train_sizes_abs[-1]/len(X))*100:.1f}%)")

    # Check if sklearn limited the max train size (common for stratified classification)
    max_requested = max(train_sizes) if isinstance(train_sizes[0], float) else max(train_sizes)/len(X)
    max_actual = train_sizes_abs[-1] / len(X)
    if max_actual < 0.9 and max_requested >= 0.9:
        print(f"  [NOTE] sklearn limited max train_size to {max_actual*100:.1f}% due to stratified CV constraints.")
        print(f"         This is expected for classification with cv={cv}. Curves are still diagnostic.")

    print(f"  Final training score: {train_scores_mean[-1]:.4f} (±{train_scores_std[-1]:.4f})")
    print(f"  Final CV score:       {test_scores_mean[-1]:.4f} (±{test_scores_std[-1]:.4f})")
    print(f"  Score gap (overfitting): {train_scores_mean[-1] - test_scores_mean[-1]:.4f}")

    # Store statistics for summary table
    learning_curve_stats = {
        'model_name': model_name,
        'model_type': model_type,
        'search_type': search_type,  # Include search_type for comparison
        'cv_folds': cv,  # Number of CV folds used
        'train_score': train_scores_mean[-1],
        'train_std': train_scores_std[-1],
        'cv_score': test_scores_mean[-1],
        'cv_std': test_scores_std[-1],
        'gap': train_scores_mean[-1] - test_scores_mean[-1]
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training scores
    ax.plot(train_sizes_abs, train_scores_mean, 'o-', color='#d62728',
            label='Training score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes_abs,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.15, color='#d62728')

    # Plot cross-validation scores in the same plot to compare
    ax.plot(train_sizes_abs, test_scores_mean, 'o-', color='#2ca02c',
            label='Cross-validation score', linewidth=2, markersize=8)
    ax.fill_between(train_sizes_abs,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.15, color='#2ca02c')

    # Customize plot with percentage labels on x-axis
    # Calculate percentages for each training size
    max_samples = len(X)
    percentages = [(size / max_samples) * 100 for size in train_sizes_abs]

    # Create custom x-tick labels showing both count and percentage
    # Format: "N (P%)" where N is sample count and P is percentage
    xtick_labels = [f'{int(size)}\n({pct:.0f}%)' for size, pct in zip(train_sizes_abs, percentages)]

    # Set x-ticks and labels
    ax.set_xticks(train_sizes_abs)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel('Training Examples (Percentage of Training Set)', fontsize=12, fontweight='bold')

    # Y-label based on scoring metric
    if scoring == 'accuracy':
        ylabel = 'Accuracy Score'
    elif scoring in ['f1', 'f1_weighted', 'f1_macro', 'f1_micro']:
        ylabel = 'F1 Score'
    elif scoring == 'r2':
        ylabel = 'R² Score'
    elif 'mean_squared_error' in scoring:
        ylabel = 'Negative MSE'
    elif 'mean_absolute_error' in scoring:
        ylabel = 'Negative MAE'
    else:
        ylabel = 'Score'

    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Title with model info (include search_type)
    search_label = 'GridSearchCV' if search_type == 'grid' else 'RandomizedSearchCV'
    title = f'Learning Curve: {model_name} ({search_label})\n'
    title += f'{model_type.capitalize()} - {cv}-Fold CV - Scoring: {scoring}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Legend
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)

    # Add diagnostic text box
    # get the gap between training and test scores at max training size
    gap = train_scores_mean[-1] - test_scores_mean[-1]
    if gap > 0.1:
        diagnosis = "⚠ High variance (overfitting)\nConsider: regularization, more data"
        box_color = '#ffcccc'
    elif test_scores_mean[-1] < 0.7:
        diagnosis = "⚠ High bias (underfitting)\nConsider: more features, complex model"
        box_color = '#ffffcc'
    else:
        diagnosis = "✓ Good fit\nModel generalizes well"
        box_color = '#ccffcc'

    props = dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='gray', linewidth=1)
    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save figure if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Include search_type and CV in filename for comparison
        filename = f'{model_name}_{model_type}_{search_type}_cv{cv}_learning_curve.png'
        save_path = output_path / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n[SAVED] Learning curve ({search_type}) saved to: {save_path}")
        print("="*80 + "\n")

        return save_path, learning_curve_stats
    else:
        plt.show()
        return None, learning_curve_stats
