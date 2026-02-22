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
import traceback
from pathlib import Path

# Use non-interactive backend to avoid tkinter "main thread" errors
import matplotlib
matplotlib.use('Agg')

from core.models import ModelTrainer
from visualization.analysis import plot_feature_distributions
from visualization.classification import plot_confusion_matrix, plot_percentage_confusion_matrix
from visualization.cv_analysis import generate_cv_analysis_report

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import time
import pandas as pd

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

from config import Config
from core.preprocessing import DataPreprocessor
from core.tuning import ModelTuner
from core.metrics import MetricsCollector

import argparse



def prepare_data(test_classification=True):
    """
    Prepare data once for all experiments.

    This function handles all preprocessing including:
    - Loading data
    - Creating target variables (with class finding if needed)
    - Train-test split
    - Feature scaling

    Args:
        test_classification: Whether classification will be tested (triggers class finding)

    Returns:
        Dictionary with prepared data for classification and regression
    """
    print("\n" + "="*80)
    print(" "*20 + "DATA PREPARATION")
    print("="*80)

    # Load and prepare data
    preprocessor = DataPreprocessor(
        data_path=Config.DATA_PATH,
        aggregation_cols=Config.AGGREGATION_COLS,
        aggregation_name=Config.AGGREGATION_NAME
    )
    preprocessor.load_data()

    # Remove outliers using IQR method (if enabled in Config)
    if Config.REMOVE_OUTLIERS:
        preprocessor.remove_outliers_iqr(multiplier=Config.OUTLIER_IQR_MULTIPLIER)

    # Create target variables - class finding happens here (only once!)
    preprocessor.create_target_variables(
        n_classes=Config.N_CLASSES,
        clustering_method=Config.CLUSTERING_METHOD,
        create_classes=test_classification
    )

    data = preprocessor.get_feature_target_split()
    X = data['X']
    y_classification = data['y_classification']
    y_regression = data['y_regression']

    # Get actual number of classes (important when N_CLASSES='auto')
    actual_n_classes = len(y_classification.unique()) if test_classification else 0
    print(f"Actual number of classes: {actual_n_classes}")

    # Classification train-test split
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_classification,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    # Standardize features for classification
    scaler_clf = StandardScaler()
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)

    # Regression train-test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    # Scale regression data separately
    feature_names_reg = X_train_reg.columns.tolist() if hasattr(X_train_reg, 'columns') else None
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Convert back to DataFrame to preserve feature names (avoids LightGBM warnings)
    if feature_names_reg:
        X_train_reg_scaled = pd.DataFrame(X_train_reg_scaled, columns=feature_names_reg)
        X_test_reg_scaled = pd.DataFrame(X_test_reg_scaled, columns=feature_names_reg)

    # Prepare polynomial features for linear regression models (if enabled)
    X_train_reg_poly = None
    X_test_reg_poly = None
    if Config.USE_POLYNOMIAL_FEATURES:
        degree = Config.POLYNOMIAL_DEGREE
        print(f"\nGenerating polynomial features (degree={degree})...")
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_reg_poly = poly.fit_transform(X_train_reg_scaled)
        X_test_reg_poly = poly.transform(X_test_reg_scaled)
        poly_feature_names = poly.get_feature_names_out(feature_names_reg) if feature_names_reg else None
        X_train_reg_poly = pd.DataFrame(X_train_reg_poly, columns=poly_feature_names)
        X_test_reg_poly = pd.DataFrame(X_test_reg_poly, columns=poly_feature_names)
        print(f"  Original features: {X_train_reg_scaled.shape[1]}")
        print(f"  Polynomial features: {X_train_reg_poly.shape[1]}")

    print("\nData preparation complete!")
    print("-"*80)

    return {
        # Classification data
        'X_train_clf': X_train_clf_scaled,
        'X_test_clf': X_test_clf_scaled,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        # Regression data (original scaled)
        'X_train_reg': X_train_reg_scaled,
        'X_test_reg': X_test_reg_scaled,
        # Regression data (polynomial - for linear models only)
        'X_train_reg_poly': X_train_reg_poly,
        'X_test_reg_poly': X_test_reg_poly,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        # Metadata
        'actual_n_classes': actual_n_classes
    }


def run_cv_experiment(prepared_data, cv_values=[3, 5], model_name='RandomForest',
                      test_classification=True, test_regression=True):
    """
    Run experiment with different CV fold values using pre-prepared data.

    Args:
        prepared_data: Dictionary with prepared data from prepare_data()
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

    # Extract prepared data
    X_train = prepared_data['X_train_clf']
    X_test = prepared_data['X_test_clf']
    y_train_clf = prepared_data['y_train_clf']
    y_test_clf = prepared_data['y_test_clf']
    X_train_reg = prepared_data['X_train_reg']
    X_test_reg = prepared_data['X_test_reg']
    X_train_reg_poly = prepared_data.get('X_train_reg_poly')
    X_test_reg_poly = prepared_data.get('X_test_reg_poly')
    y_train_reg = prepared_data['y_train_reg']
    y_test_reg = prepared_data['y_test_reg']
    actual_n_classes = prepared_data['actual_n_classes']

    # Models that benefit from polynomial features (linear models only)
    LINEAR_MODELS = {'Ridge', 'Lasso', 'ElasticNet', 'LinearRegression'}

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

            # Generate confusion matrices (raw counts + 3 percentage normalizations)
            agg_label = f'{Config.SEARCH_TYPE}_cv{cv}'
            target_classes = sorted(y_test_clf.unique())
            try:
                plot_confusion_matrix(
                    y_test_clf, y_pred,
                    target_classes=target_classes,
                    model_name=model_name,
                    output_path=Config.CLF_TUNED_CONF_MATRIX_DIR,
                    custom_aggregation_name=agg_label
                )
                for normalize in ('total', 'true', 'pred'):
                    plot_percentage_confusion_matrix(
                        y_test_clf, y_pred,
                        target_classes=target_classes,
                        model_name=model_name,
                        output_path=Config.CLF_TUNED_CONF_MATRIX_DIR,
                        normalize=normalize,
                        custom_aggregation_name=agg_label
                    )
            except Exception as e:
                print(f"  [WARNING] Confusion matrix generation failed: {e}")
                traceback.print_exc()

            # Generate learning curve for each CV value to compare convergence
            generate_learning_curve_for_model(
                model=best_model,
                X_train=X_train,
                y_train=y_train_clf,
                model_name=model_name,
                task_type='classification',
                cv=cv
            )

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

            # Use polynomial features for linear models, original for tree-based
            use_poly = (reg_model_name in LINEAR_MODELS
                        and X_train_reg_poly is not None)
            X_tr = X_train_reg_poly if use_poly else X_train_reg
            X_te = X_test_reg_poly if use_poly else X_test_reg

            if use_poly:
                print(f"  Using polynomial features ({X_tr.shape[1]} features)")
            else:
                print(f"  Using original features ({X_tr.shape[1]} features)")

            best_model, best_params, best_score = tuner.tune_model(
                X_tr, y_train_reg,
                model_type='regression',
                model_name=reg_model_name,
                search_type=Config.SEARCH_TYPE,
                cv=cv
            )

            tuning_time = time.time() - start_time

            # Evaluate on test set
            y_pred = best_model.predict(X_te)

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

            # Generate learning curve for each CV value to compare convergence
            generate_learning_curve_for_model(
                model=best_model,
                X_train=X_tr,
                y_train=y_train_reg,
                model_name=reg_model_name,
                task_type='regression',
                cv=cv
            )

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
        print(clf_results[['Model', 'CV_Folds', 'Tuning_Time_sec', 'Best_CV_Score', 'Test_Accuracy', 'Test_F1']].to_string(index=False))

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
        print(reg_results[['Model', 'CV_Folds', 'Tuning_Time_sec', 'Best_CV_Score', 'Test_R2', 'Test_RMSE']].to_string(index=False))

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

# Shared trainer for learning curve collection across all models
_learning_curve_trainer = None

def get_learning_curve_trainer():
    """Get or create the shared ModelTrainer for learning curve collection"""
    global _learning_curve_trainer
    if _learning_curve_trainer is None:
        metrics_collector = MetricsCollector()
        _learning_curve_trainer = ModelTrainer(metrics_collector=metrics_collector, output_config=Config)
    return _learning_curve_trainer

def generate_learning_curve_for_model(model, X_train, y_train, model_name, task_type, cv):
    """Generate learning curve for a trained model and collect stats.

    For classification: uses Config.CLF_LEARNING_CURVES_SCORE ('accuracy', 'f1', or 'both')
    For regression: uses Config.REG_LEARNING_CURVES_SCORE ('r2', 'rmse', 'mae', or 'both')
    """
    if not Config.GENERATE_LEARNING_CURVES:
        return

    trainer = get_learning_curve_trainer()

    # Determine scoring metric(s) based on task type
    if task_type == 'classification':
        score_setting = getattr(Config, 'CLF_LEARNING_CURVES_SCORE', 'accuracy')
        if score_setting == 'both':
            scoring_list = ['accuracy', 'f1_weighted']
        elif score_setting == 'f1':
            scoring_list = ['f1_weighted']
        else:
            scoring_list = ['accuracy']
    else:
        # Regression scoring from config
        reg_score_setting = getattr(Config, 'REG_LEARNING_CURVES_SCORE', 'r2')
        if reg_score_setting == 'both':
            scoring_list = ['r2', 'neg_root_mean_squared_error']
        elif reg_score_setting == 'rmse':
            scoring_list = ['neg_root_mean_squared_error']
        elif reg_score_setting == 'mae':
            scoring_list = ['neg_mean_absolute_error']
        else:
            scoring_list = ['r2']

    for scoring in scoring_list:
        trainer._generate_learning_curve(
            model=model,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            model_type=task_type,
            search_type=Config.SEARCH_TYPE,
            cv=cv,
            scoring=scoring
        )

def generate_learning_curves_summary():
    """Generate summary table from all collected learning curve stats"""
    if not Config.GENERATE_LEARNING_CURVES:
        print("Learning curve generation is disabled in Config. Skipping...")
        return

    print("\nGenerate Learning Curves Summary Table")
    print("-" * 80)
    trainer = get_learning_curve_trainer()
    trainer.generate_learning_curves_summary()




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

    # Determine if we need classification classes
    test_classification = task_type in ['classification', 'both']
    test_regression = task_type in ['regression', 'both']

    # =========================================================================
    # PREPARE DATA ONCE (including class finding if classification is requested)
    # =========================================================================
    prepared_data = prepare_data(test_classification=test_classification)

    # Test classification models
    if test_classification:
        print("\n" + "="*80)
        print("CLASSIFICATION MODELS EXPERIMENTS")
        print("="*80)

        for i, model_name in enumerate(classification_models, 1):
            print(f"\n[{i}/{len(classification_models)}] Testing {model_name}...")
            try:
                df_model = run_cv_experiment(
                    prepared_data=prepared_data,  # Pass pre-prepared data
                    cv_values=cv_values_to_test,
                    model_name=model_name,
                    test_classification=True,
                    test_regression=False
                )
                all_results.append(df_model)
                analyze_results(df_model)
                save_results(df_model, f'./output/reports/cv_experiment_{model_name}_{clustering_method}_classification.xlsx')
            except Exception as e:
                print(f"  Error testing {model_name}: {e}")
                traceback.print_exc()
                continue

    # Test regression models
    if test_regression:
        print("\n" + "="*80)
        print("REGRESSION MODELS EXPERIMENTS")
        print("="*80)

        for i, model_name in enumerate(regression_models, 1):
            print(f"\n[{i}/{len(regression_models)}] Testing {model_name}...")
            try:
                df_model = run_cv_experiment(
                    prepared_data=prepared_data,  # Pass pre-prepared data
                    cv_values=cv_values_to_test,
                    model_name=model_name,
                    test_classification=False,
                    test_regression=True
                )
                all_results.append(df_model)
                analyze_results(df_model)
                save_results(df_model, f'./output/reports/cv_experiment_{model_name}_{clustering_method}_regression.xlsx')
            except Exception as e:
                print(f"  Error testing {model_name}: {e}")
                traceback.print_exc()
                continue

    # Combined results
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        save_results(df_all, f'./output/reports/cv_experiment_all_models_{clustering_method}.xlsx')

        print("\n" + "="*80)
        print("SUMMARY OF ALL MODELS")
        print("="*80)
        analyze_results(df_all)

        # Generate CV analysis visualizations
        print("\n" + "="*80)
        print("GENERATING CV ANALYSIS VISUALIZATIONS")
        print("="*80)

        cv_graphs_dir = Config.OUTPUT_DIR / 'graphs' / 'cv_analysis'
        dataset_info = Config.get_dataset_info()

        # Generate visualizations for classification
        if task_type in ['classification', 'both']:
            clf_results = df_all[df_all['Task'] == 'Classification']
            if not clf_results.empty:
                generate_cv_analysis_report(
                    clf_results,
                    cv_graphs_dir / 'classification',
                    task_type='classification',
                    dataset_info=dataset_info
                )

        # Generate visualizations for regression
        if task_type in ['regression', 'both']:
            reg_results = df_all[df_all['Task'] == 'Regression']
            if not reg_results.empty:
                generate_cv_analysis_report(
                    reg_results,
                    cv_graphs_dir / 'regression',
                    task_type='regression',
                    dataset_info=dataset_info
                )

        print(f"\nAll CV analysis graphs saved to: {cv_graphs_dir}")

        # Generate learning curves summary (individual curves were generated during training)
        generate_learning_curves_summary()
      
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("\nBased on the results above:")
    print("1. Check if performance is stable across CV values (low std dev)")
    print("2. If stable: Use CV=3 for faster training with minimal performance loss")
    print("3. If variable: Use CV=5 for better balance of speed and reliability")
    print("4. For final model selection: You may consider increasing to CV=10 if needed")
    print("\n" + "="*80 + "\n")
