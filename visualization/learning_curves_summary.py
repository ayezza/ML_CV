"""
Learning Curves Summary Table Generator

This module creates summary tables aggregating learning curve statistics
for all trained models.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def create_learning_curves_summary_table(stats_list, output_path=None):
    """
    Create a summary table of learning curve statistics for all models.

    Args:
        stats_list: List of dictionaries containing learning curve statistics
                   Each dict should have: model_name, model_type, train_score,
                   train_std, cv_score, cv_std, gap
        output_path: Directory path to save the summary table image (optional)

    Returns:
        Path to saved table image (if output_path provided), or None

    Example:
        >>> stats = [
        ...     {'model_name': 'RandomForest', 'model_type': 'classification',
        ...      'train_score': 0.995, 'train_std': 0.003,
        ...      'cv_score': 0.946, 'cv_std': 0.019, 'gap': 0.049},
        ...     ...
        ... ]
        >>> create_learning_curves_summary_table(stats, 'output/graphs/learning_curves/')
    """
    if not stats_list:
        print("[WARNING] No learning curve statistics to summarize")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(stats_list)

    # Add diagnosis column based on gap and scores
    def diagnose(row):
        gap = row['gap']
        train_score = row['train_score']
        cv_score = row['cv_score']

        if gap > 0.10:
            return "⚠️ High Variance (Overfitting)"
        elif train_score < 0.75 and cv_score < 0.70:
            return "⚠️ High Bias (Underfitting)"
        elif gap < 0.05 and train_score > 0.85 and cv_score > 0.80:
            return "✓ Good Fit"
        else:
            return "Moderate Fit"

    df['diagnosis'] = df.apply(diagnose, axis=1)

    # Separate classification and regression
    clf_df = df[df['model_type'] == 'classification'].copy()
    reg_df = df[df['model_type'] == 'regression'].copy()

    # Create figure with two subplots (classification and regression)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, max(10, len(df) * 0.4)))
    fig.suptitle('Learning Curves Summary - All Models', fontsize=16, fontweight='bold', y=0.995)

    # Function to create table
    def create_table(ax, data, title):
        if data.empty:
            ax.text(0.5, 0.5, f'No {title} models', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data (include Search Type column)
        table_data = []
        headers = ['Model', 'Search', 'Train Score', 'CV Score', 'Gap', 'Diagnosis']

        for _, row in data.iterrows():
            # Get search_type, default to 'grid' if not present (for backward compatibility)
            search_type = row.get('search_type', 'grid')
            search_label = 'Grid' if search_type == 'grid' else 'Random'

            table_data.append([
                row['model_name'],
                search_label,
                f"{row['train_score']:.4f} (±{row['train_std']:.4f})",
                f"{row['cv_score']:.4f} (±{row['cv_std']:.4f})",
                f"{row['gap']:.4f}",
                row['diagnosis']
            ])

        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='left', loc='center',
                        bbox=[0, 0, 1, 1])

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)

        # Header style
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white', fontsize=10)
            cell.set_edgecolor('white')
            cell.set_linewidth(2)

        # Row colors based on diagnosis
        for i, (_, row) in enumerate(data.iterrows(), start=1):
            # Alternate row colors
            if i % 2 == 0:
                base_color = '#F2F2F2'
            else:
                base_color = 'white'

            # Color based on diagnosis
            if "Good Fit" in row['diagnosis']:
                diag_color = '#C6EFCE'  # Light green
            elif "Overfitting" in row['diagnosis']:
                diag_color = '#FFEB9C'  # Light yellow
            elif "Underfitting" in row['diagnosis']:
                diag_color = '#FFC7CE'  # Light red
            else:
                diag_color = base_color

            # Apply colors
            for j in range(len(headers)):
                cell = table[(i, j)]
                if j == 4:  # Diagnosis column
                    cell.set_facecolor(diag_color)
                else:
                    cell.set_facecolor(base_color)
                cell.set_edgecolor('lightgray')
                cell.set_linewidth(0.5)

                # Bold model name
                if j == 0:
                    cell.set_text_props(weight='bold')

    # Create classification table
    create_table(ax1, clf_df, 'Classification Models')

    # Create regression table
    create_table(ax2, reg_df, 'Regression Models')

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        save_path = output_path / 'learning_curves_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print("\n" + "="*80)
        print("LEARNING CURVES SUMMARY TABLE")
        print("="*80)
        print(f"[SAVED] Summary table saved to: {save_path}")
        print("="*80 + "\n")

        # Also print text summary
        print("\nCLASSIFICATION MODELS:")
        print("-" * 80)
        if not clf_df.empty:
            print(f"{'Model':<25} {'Train Score':<20} {'CV Score':<20} {'Gap':<10} {'Diagnosis'}")
            print("-" * 80)
            for _, row in clf_df.iterrows():
                print(f"{row['model_name']:<25} "
                      f"{row['train_score']:.4f} (±{row['train_std']:.4f})"
                      f"  {row['cv_score']:.4f} (±{row['cv_std']:.4f})"
                      f"  {row['gap']:<8.4f}  {row['diagnosis']}")
        else:
            print("No classification models")

        print("\n\nREGRESSION MODELS:")
        print("-" * 80)
        if not reg_df.empty:
            print(f"{'Model':<25} {'Train Score':<20} {'CV Score':<20} {'Gap':<10} {'Diagnosis'}")
            print("-" * 80)
            for _, row in reg_df.iterrows():
                print(f"{row['model_name']:<25} "
                      f"{row['train_score']:.4f} (±{row['train_std']:.4f})"
                      f"  {row['cv_score']:.4f} (±{row['cv_std']:.4f})"
                      f"  {row['gap']:<8.4f}  {row['diagnosis']}")
        else:
            print("No regression models")

        print("\n" + "="*80 + "\n")

        return save_path
    else:
        plt.show()
        return None
