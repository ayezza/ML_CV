"""
Classification Visualization Module

This module provides visualization functions for classification models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from pathlib import Path


def plot_confusion_matrix(y_test, y_pred, target_classes, model_name,
                          output_path, is_tuned=False,
                          custom_aggregation_name='sum', title=None):
    """
    Generate and save confusion matrix visualization

    Args:
        y_test: True labels
        y_pred: Predicted labels
        target_classes: List of class labels
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        custom_aggregation_name: Name of aggregation function used
        title: Custom title (optional)

    Returns:
        Path: Full path to the saved confusion matrix plot

    Example:
        >>> plot_confusion_matrix(
        ...     y_test, y_pred,
        ...     target_classes=[0, 1, 2, 3],
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/classification/base/confusion_matrices'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    targets = sorted(target_classes)

    print("="*80)
    print(" "*25 + "CONFUSION MATRIX")
    print("="*80)

    # 1. Compute confusion matrix with sklearn
    cm = confusion_matrix(y_test, y_pred, labels=target_classes)
    print("\nConfusion matrix as array with Sklearn:")
    print(cm)

    # 2. Create DataFrame for better display
    cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
    print("\nConfusion matrix as DataFrame with Pandas:")
    print(cm_df)

    # 3. Create heatmap with seaborn
    print("\nGenerating confusion matrix visualization...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(data=cm, annot=True, cmap='Blues',
                xticklabels=targets, yticklabels=targets,
                cbar=False, square=True, fmt=".0f")
    plt.xlabel('Real classes', fontsize=12)
    plt.ylabel('Predicted classes', fontsize=12)

    if title is None:
        title = f'Confusion Matrix - {model_name}'

    plt.title(f"{title}\n(Aggregation: {custom_aggregation_name})", fontsize=14)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f"CONF_MATRIX_{model_name.replace(' ', '_').lower()}_{custom_aggregation_name}.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_roc_curve(y_test, y_score, model_name, output_path,
                   is_tuned=False, n_classes=4,
                   custom_aggregation_name='sum', title=None):
    """
    Generate and save multiclass ROC curve using One-vs-Rest approach

    Args:
        y_test: True labels
        y_score: Prediction probabilities (shape: n_samples x n_classes)
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        n_classes: Number of classes (default: 4)
        custom_aggregation_name: Name of aggregation function used
        title: Custom title (optional)

    Returns:
        dict: Dictionary containing fpr, tpr, and roc_auc for each class

    Example:
        >>> plot_roc_curve(
        ...     y_test, y_score,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/classification/base/roc_curves'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*25 + "ROC CURVE (MULTICLASS)")
    print("="*80)

    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        plt.fill_between(fpr[i], tpr[i], alpha=0.1, color=color)

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    if title is None:
        title = f'ROC Curve (Multiclass) - {model_name}'

    plt.title(f'{title}\n(Aggregation: {custom_aggregation_name})', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f"ROC_Curve_Multiclass_{model_name.replace(' ', '_')}_{custom_aggregation_name}.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMicro-average AUC-ROC for {model_name}: {roc_auc['micro']:.5f}")
    for i in range(n_classes):
        print(f"  Class {i} AUC: {roc_auc[i]:.5f}")

    print(f"\n✓ ROC curve saved to: {save_path}")
    print("="*80 + "\n")

    return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


def plot_classification_bar_chart(y_test, y_pred, model_name, output_path,
                                  is_tuned=False, custom_aggregation_name='sum',
                                  title=None):
    """
    Generate bar chart comparing actual vs predicted class distributions

    Args:
        y_test: True class labels
        y_pred: Predicted class labels
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        custom_aggregation_name: Name of aggregation function used
        title: Custom title (optional)

    Returns:
        Path: Full path to the saved bar chart

    Example:
        >>> plot_classification_bar_chart(
        ...     y_test, y_pred,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/classification/base/bar_charts'),
        ...     custom_aggregation_name='manhattan'
        ... )
    """
    print("="*80)
    print(" "*20 + "CLASSIFICATION BAR CHART")
    print("="*80)

    plt.figure(figsize=(12, 7))

    # Get unique classes and count distributions
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

    # Count actual and predicted for each class
    actual_counts = []
    predicted_counts = []

    for cls in classes:
        actual_counts.append(np.sum(y_test == cls))
        predicted_counts.append(np.sum(y_pred == cls))

    # Set up bar positions
    x = np.arange(len(classes))
    width = 0.35

    # Create bars
    bars1 = plt.bar(x - width/2, actual_counts, width, label='Actual',
                    alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = plt.bar(x + width/2, predicted_counts, width, label='Predicted',
                    alpha=0.8, color='coral', edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Add accuracy as text box
    textstr = f'Accuracy = {accuracy:.4f}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=props, fontweight='bold')

    plt.xlabel('Class Label', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')

    if title is None:
        title = f'Class Distribution: Actual vs Predicted - {model_name}'

    plt.title(f'{title}\n(Aggregation: {custom_aggregation_name})', fontsize=14, fontweight='bold')
    plt.xticks(x, [f'Class {cls}' for cls in classes])
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f'classification_bar_{model_name}_{custom_aggregation_name}.png'
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Classification bar chart saved to: {save_path}")
    print("="*80 + "\n")

    return save_path


def plot_probability_matrix(model, X_test, y_test, model_name, output_path,
                           is_tuned=False, n_samples_display=10):
    """
    Generate probability matrix heatmap showing prediction probabilities

    Args:
        model: Trained classification model with predict_proba method
        X_test: Test features
        y_test: True test labels
        model_name: Name of the model for display
        output_path: Directory path where plot will be saved
        is_tuned: Whether this is for a tuned model (affects subdirectory)
        n_samples_display: Number of samples to visualize in heatmap (default: 10)

    Returns:
        tuple: (save_path, prob_df) - Path to saved plot and DataFrame with probabilities

    Example:
        >>> plot_probability_matrix(
        ...     model, X_test, y_test,
        ...     model_name='RandomForest',
        ...     output_path=Path('./output/graphs/classification/base/probability_matrices'),
        ...     n_samples_display=20
        ... )
    """
    print("="*80)
    print(" "*25 + "PROBABILITY MATRIX")
    print("="*80)

    # Get probability predictions
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Get class labels
    classes = model.classes_ if hasattr(model, 'classes_') else sorted(set(y_test))

    # Create DataFrame with probabilities
    prob_df = pd.DataFrame(y_proba, columns=[f'Prob_Class_{cls}' for cls in classes])
    prob_df['Predicted_Class'] = y_pred
    prob_df['True_Class'] = y_test.values if hasattr(y_test, 'values') else y_test
    prob_df['Correct'] = prob_df['Predicted_Class'] == prob_df['True_Class']

    # Add prediction confidence (max probability)
    prob_df['Confidence'] = y_proba.max(axis=1)

    # Reorder columns for better readability
    first_cols = ['True_Class', 'Predicted_Class', 'Correct', 'Confidence']
    prob_cols = [col for col in prob_df.columns if col.startswith('Prob_Class_')]
    prob_df = prob_df[first_cols + prob_cols]

    # Display summary statistics
    print(f"\nTotal samples: {len(prob_df)}")
    print(f"Correct predictions: {prob_df['Correct'].sum()} ({prob_df['Correct'].mean()*100:.2f}%)")
    print(f"Average confidence: {prob_df['Confidence'].mean():.4f}")

    # Create visualization: Heatmap of probabilities for sample predictions
    plt.figure(figsize=(12, 8))

    # Select samples to visualize
    n_viz = min(n_samples_display, len(prob_df))
    viz_samples = prob_df.head(n_viz)

    # Extract probability values for heatmap
    prob_matrix = viz_samples[[col for col in viz_samples.columns
                               if col.startswith('Prob_Class_')]].values

    # Create heatmap
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[f'Class {cls}' for cls in classes],
                yticklabels=[f"Sample {i}\n(T:{t}, P:{p})"
                           for i, (t, p) in enumerate(zip(viz_samples['True_Class'],
                                                          viz_samples['Predicted_Class']))],
                cbar_kws={'label': 'Probability'})
    plt.title(f'Probability Matrix - {model_name}\n(First {n_viz} samples)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('Test Samples', fontsize=12)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save figure
    filename = f"probability_heatmap_{model_name.replace(' ', '_')}.png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Probability heatmap saved to: {save_path}")
    print("="*80 + "\n")

    return save_path, prob_df


# Legacy function names for backward compatibility
def generate_conf_matrix(y_test, y_pred, target, model_name, title,
                        custom_aggregation_name='sum', output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    return plot_confusion_matrix(y_test, y_pred, target, model_name,
                                output_path, custom_aggregation_name=custom_aggregation_name,
                                title=title)


def generate_auc_roc_curve_multiclass(model_name, y_test, y_score, n_classes=4,
                                      custom_aggregation_name='sum', output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    return plot_roc_curve(y_test, y_score, model_name, output_path,
                         n_classes=n_classes,
                         custom_aggregation_name=custom_aggregation_name)


def generate_classification_bar_chart(y_test, y_pred, model_name,
                                     title="Class Distribution: Actual vs Predicted",
                                     custom_aggregation_name='sum', output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    return plot_classification_bar_chart(y_test, y_pred, model_name, output_path,
                                        custom_aggregation_name=custom_aggregation_name,
                                        title=title)


def generate_probability_matrix(model, X_test, y_test, model_name,
                               n_samples=10, save_full=True, output_path=None):
    """Legacy function for backward compatibility"""
    if output_path is None:
        output_path = Path('.')
    save_path, prob_df = plot_probability_matrix(model, X_test, y_test, model_name,
                                                 output_path, n_samples_display=n_samples)
    return prob_df
