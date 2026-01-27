"""
Cross-Validation Fold Experiment

This script tests different CV fold values to see their impact on:
- Hyperparameter tuning time
- Model performance consistency
- Best parameter selection

Usage:
    # Test with default CV values (3, 5) and all models
    python cv_experiment.py

    # Test with custom CV values
    python cv_experiment.py --cv 3 5 10

    # Test specific models only
    python cv_experiment.py --models SVC RandomForest LogisticRegression

    # Test specific models with custom CV values
    python cv_experiment.py --models SVC --cv 3 5 10

    # Test only classification models
    python cv_experiment.py --task classification --cv 3 5

    # Test only regression models
    python cv_experiment.py --task regression --cv 5 10

    # Combine all options
    python cv_experiment.py --models SVC RandomForest --cv 3 5 10 --task both
"""
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import time
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import Config
from core.preprocessing import DataPreprocessor
from core.tuning import ModelTuner
from core.metrics import MetricsCollector

import argparse



def run_cv_experiment(cv_values=[3, 5], model_name='RandomForest', test_classification=True, test_regression=True):
    """
    Run experiment with different CV fold values

    Args:
        cv_values: List of CV fold values to test
        model_name: Model to test ('RandomForest', 'SVC', etc.)
        test_classification: Whether to test classification (default: True)
        test_regression: Whether to test regression (default: True)

    Returns:
        DataFrame with results
    """
    print("\n" + "="*80)
    print(" "*20 + f"CV FOLD EXPERIMENT - {model_name}")
    print("="*80)
    print(f"\nTesting CV values: {cv_values}")
    print(f"Model: {model_name}")
    print("-"*80)

    # Load and prepare data
    preprocessor = DataPreprocessor(
        data_path=Config.DATA_PATH,
        aggregation_cols=Config.AGGREGATION_COLS,
        aggregation_name=Config.AGGREGATION_NAME
    )
    preprocessor.load_data()
    # preprocessor.(aggregation_name=Config.AGGREGATION_NAME)
    preprocessor.create_target_variables(n_classes=Config.N_CLASSES, clustering_method=Config.CLUSTERING_METHOD)

    data = preprocessor.get_feature_target_split()
    X = data['X']
    y_classification = data['y_classification']
    y_regression = data['y_regression']


    # ClassificationTrain-test split
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(
        X, y_classification,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    # Get actual number of classes (important when N_CLASSES='auto')
    actual_n_classes = len(y_classification.unique())
    print(f"Actual number of classes: {actual_n_classes}")

    # REGRESSION Standardize features by scaling to allow models like SVM to perform better (using Gradient Descent algorithms)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    results = []

    # Test classification
    if test_classification:
        print("\n" + "="*80)
        print("CLASSIFICATION EXPERIMENTS")
        print("="*80)

        for cv in cv_values:
            print(f"\n--- Testing CV={cv} for Classification ---")

            # Choose tuning strategy: 'grid' or 'random' as per Config
            tuner = ModelTuner(random_state=Config.RANDOM_STATE)
            metrics_collector = MetricsCollector()

            # Time the tuning process
            start_time = time.time()

            best_model, best_params, best_score = tuner.tune_model(
                X_train, y_train_clf,
                model_type='classification',
                model_name=model_name,
                search_type=Config.SEARCH_TYPE,
                cv=cv
            )

            tuning_time = time.time() - start_time

            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            # Get predicted probabilities if available (for classifiers that support it)
            y_score = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

            test_metrics = metrics_collector.collect_classification_metrics(
                y_test_clf, y_pred, y_score, n_classes=actual_n_classes
            )

            results.append({
                'Task': 'Classification',
                'Model': model_name,
                'CV_Folds': cv,
                'Tuning_Time_sec': tuning_time,
                'Best_CV_Score': best_score,
                'Test_Accuracy': test_metrics['Accuracy'],
                'Test_F1': test_metrics['F1_Macro'],
                'Test_AUC': test_metrics['AUC_ROC_Micro'],
                'Best_Params': str(best_params)
            })

            print(f"  Tuning time: {tuning_time:.2f} seconds")
            print(f"  Best CV score: {best_score:.5f}")
            print(f"  Test accuracy: {test_metrics['Accuracy']:.5f}")
            print(f"  Best params: {best_params}")

    # Test regression
    if test_regression:
        print("\n" + "="*80)
        print("REGRESSION EXPERIMENTS")
        print("="*80)

        for cv in cv_values:
            print(f"\n--- Testing CV={cv} for Regression ---")
            # Choose tuning strategy: 'grid' or 'random' as per Config
            tuner = ModelTuner(random_state=Config.RANDOM_STATE)
            metrics_collector = MetricsCollector()

            # Time the tuning process
            start_time = time.time()

            # Map model names for regression
            reg_model_name = 'SVM' if model_name == 'SVC' else model_name

            best_model, best_params, best_score = tuner.tune_model(
                X_train_reg, y_train_reg,
                model_type='regression',
                model_name=reg_model_name,
                search_type=Config.SEARCH_TYPE,
                cv=cv
            )

            tuning_time = time.time() - start_time

            # Evaluate on test set
            y_pred = best_model.predict(X_test_reg)

            test_metrics = metrics_collector.collect_regression_metrics(
                y_test_reg, y_pred
            )

            results.append({
                'Task': 'Regression',
                'Model': reg_model_name,
                'CV_Folds': cv,
                'Tuning_Time_sec': tuning_time,
                'Best_CV_Score': best_score,
                'Test_R2': test_metrics['R2_Score'],
                'Test_RMSE': test_metrics['RMSE'],
                'Test_MAE': test_metrics['MAE'],
                'Best_Params': str(best_params)
            })

            print(f"  Tuning time: {tuning_time:.2f} seconds")
            print(f"  Best CV score: {best_score:.5f}")
            print(f"  Test R²: {test_metrics['R2_Score']:.5f}")
            print(f"  Best params: {best_params}")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    return df_results


def analyze_results(df_results):
    """
    Analyze and display experiment results

    Args:
        df_results: DataFrame with experiment results
    """
    print("\n" + "="*80)
    print(" "*30 + "ANALYSIS")
    print("="*80)

    # Classification analysis
    clf_results = df_results[df_results['Task'] == 'Classification']
    if not clf_results.empty:
        print("\nCLASSIFICATION RESULTS:")
        print("-"*80)
        print(clf_results[['CV_Folds', 'Tuning_Time_sec', 'Best_CV_Score', 'Test_Accuracy', 'Test_F1']].to_string(index=False))

        best_cv_clf = clf_results.loc[clf_results['Test_Accuracy'].idxmax()]
        print(f"\n✓ Best CV for Classification: {int(best_cv_clf['CV_Folds'])} folds")
        print(f"  Test Accuracy: {best_cv_clf['Test_Accuracy']:.5f}")
        print(f"  Tuning Time: {best_cv_clf['Tuning_Time_sec']:.2f}s")

    # Regression analysis
    reg_results = df_results[df_results['Task'] == 'Regression']
    if not reg_results.empty:
        print("\n" + "-"*80)
        print("REGRESSION RESULTS:")
        print("-"*80)
        print(reg_results[['CV_Folds', 'Tuning_Time_sec', 'Best_CV_Score', 'Test_R2', 'Test_RMSE']].to_string(index=False))

        best_cv_reg = reg_results.loc[reg_results['Test_R2'].idxmax()]
        print(f"\n✓ Best CV for Regression: {int(best_cv_reg['CV_Folds'])} folds")
        print(f"  Test R²: {best_cv_reg['Test_R2']:.5f}")
        print(f"  Tuning Time: {best_cv_reg['Tuning_Time_sec']:.2f}s")

    # Time analysis
    print("\n" + "-"*80)
    print("TIME ANALYSIS:")
    print("-"*80)
    time_comparison = df_results.groupby('CV_Folds')['Tuning_Time_sec'].mean()
    print("\nAverage tuning time by CV folds:")
    for cv, time_val in time_comparison.items():
        print(f"  CV={int(cv):2d}: {time_val:6.2f}s")

    # Speedup relative to CV=10
    if 10 in time_comparison.index:
        baseline_time = time_comparison[10]
        print("\nSpeedup relative to CV=10:")
        for cv, time_val in time_comparison.items():
            speedup = baseline_time / time_val
            print(f"  CV={int(cv):2d}: {speedup:.2f}x faster")

    # Performance stability
    print("\n" + "-"*80)
    print("PERFORMANCE STABILITY:")
    print("-"*80)

    if not clf_results.empty:
        clf_std = clf_results['Test_Accuracy'].std()
        print(f"Classification Accuracy Std Dev: {clf_std:.5f}")
        if clf_std < 0.001:
            print("  → Very stable across CV values")
        elif clf_std < 0.01:
            print("  → Moderately stable across CV values")
        else:
            print("  → Variable across CV values")

    if not reg_results.empty:
        reg_std = reg_results['Test_R2'].std()
        print(f"Regression R² Std Dev: {reg_std:.5f}")
        if reg_std < 0.001:
            print("  → Very stable across CV values")
        elif reg_std < 0.01:
            print("  → Moderately stable across CV values")
        else:
            print("  → Variable across CV values")

    print("\n" + "="*80)


def save_results(df_results, output_path='./output/cv_experiment_results.xlsx'):
    """Save experiment results to Excel"""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='All_Results', index=False)

        # Separate sheets for classification and regression
        clf_results = df_results[df_results['Task'] == 'Classification']
        reg_results = df_results[df_results['Task'] == 'Regression']

        if not clf_results.empty:
            clf_results.to_excel(writer, sheet_name='Classification', index=False)
        if not reg_results.empty:
            reg_results.to_excel(writer, sheet_name='Regression', index=False)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CV Fold Experiment - Test different CV values across models')
    parser.add_argument('--cv', nargs='+', type=int, default=[3, 5],
                        help='CV fold values to test (default: 3 5). Example: --cv 3 5 10')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to test (default: all). Example: --models SVC RandomForest')
    parser.add_argument('--task', choices=['classification', 'regression', 'both'], default='both',
                        help='Task type to test (default: both)')

    args = parser.parse_args()

    cv_values_to_test = args.cv # List of CV values to test 3, 5, etc.
    models_filter = args.models # List of models to test or None for all
    task_type = args.task # 'classification', 'regression', or 'both'

    print("\n" + "="*80)
    print(" "*20 + "CV FOLD EXPERIMENT RUNNER")
    print("="*80)
    print("\nThis experiment tests different CV fold values to determine optimal settings.")
    print(f"Testing CV values: {cv_values_to_test}")
    if models_filter:
        print(f"Testing specific models: {', '.join(models_filter)}")
    else:
        print("Testing 8 Classification models + 12 Regression models = 20 models total")
    print(f"Task type: {task_type}")
    print("-"*80)

    # Define implemented models in the core to test
    classification_models = [
        'RandomForest',
        'DecisionTree',
        'SVC',
        'KNN',
        'LogisticRegression',
        'NaiveBayes',
        'Bagging',
        'Stacking'
    ]

    regression_models = [
        'RandomForest',
        'DecisionTree',
        'SVM',
        'KNN',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'GradientBoosting',
        'XGBoost',
        'LightGBM',
        'Bagging',
        'Stacking'
    ]

    # to store all obtained results
    all_results = []

    # get clustering method from config for labeling
    clustering_method = Config.CLUSTERING_METHOD
    print(f"\nUsing clustering method for classification labels: {clustering_method}")

    # Filter models if specific models requested
    if models_filter:
        classification_models = [m for m in classification_models if m in models_filter]
        regression_models = [m for m in regression_models if m in models_filter]

    # Test classification models
    if task_type in ['classification', 'both']:
        print("\n" + "="*80)
        print("CLASSIFICATION MODELS EXPERIMENTS")
        print("="*80)

        for i, model_name in enumerate(classification_models, 1):
            print(f"\n[{i}/{len(classification_models)}] Testing {model_name}...")
            try:
                df_model = run_cv_experiment(
                    cv_values=cv_values_to_test,  # Use command-line CV values if any
                    model_name=model_name,
                    test_classification=True,
                    test_regression=False  # Classification models only
                )
                all_results.append(df_model)
                analyze_results(df_model)
                save_results(df_model, f'./output/reports/cv_experiment_{model_name}_{clustering_method}_classification.xlsx')
            except Exception as e:
                print(f"  ⚠️ Error testing {model_name}: {e}")
                continue

    # Test regression models
    if task_type in ['regression', 'both']:
        print("\n" + "="*80)
        print("REGRESSION MODELS EXPERIMENTS")
        print("="*80)

        for i, model_name in enumerate(regression_models, 1):
            print(f"\n[{i}/{len(regression_models)}] Testing {model_name}...")
            try:
                df_model = run_cv_experiment(
                    cv_values=cv_values_to_test,  # Use command-line CV values
                    model_name=model_name,
                    test_classification=False,  # Regression models only
                    test_regression=True
                )
                all_results.append(df_model)
                analyze_results(df_model)
                save_results(df_model, f'./output/reports/cv_experiment_{model_name}_{clustering_method}_regression.xlsx')
            except Exception as e:
                print(f"  ⚠️ Error testing {model_name}: {e}")
                continue

    # Combined results
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        save_results(df_all, f'./output/reports/cv_experiment_all_models_{clustering_method}.xlsx')

        print("\n" + "="*80)
        print("SUMMARY OF ALL MODELS")
        print("="*80)
        analyze_results(df_all)

    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("\nBased on the results above:")
    print("1. Check if performance is stable across CV values (low std dev)")
    print("2. If stable: Use CV=3 for faster training with minimal performance loss")
    print("3. If variable: Use CV=5 for better balance of speed and reliability")
    print("4. For final model selection: You may consider increasing to CV=10 if needed")
    print("\n" + "="*80 + "\n")
