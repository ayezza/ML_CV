# ML Cross-Validation Pipeline

A comprehensive machine learning pipeline for testing and comparing classification and regression models with cross-validation experiments. This project enables systematic evaluation of different CV fold values, hyperparameter tuning strategies, and model performance across multiple algorithms.

## Features

- **Multi-Model Comparison**: Test 8 classification and 12 regression models simultaneously
- **Cross-Validation Experiments**: Compare CV=3, 5, 10, 15 to find optimal settings for your data
- **Hyperparameter Tuning**: Support for GridSearchCV and RandomizedSearchCV with optional custom overrides
- **Automatic Preprocessing**: Handles date columns, categorical encoding, feature scaling, and outlier removal
- **Polynomial Features**: Automatic polynomial transformation for linear regression models (Ridge, Lasso, ElasticNet)
- **Multiple Aggregation Functions**: 17+ methods to combine target columns (sum, euclidean, mahalanobis, etc.)
- **Auto Class Detection**: Automatically finds optimal number of classes using clustering metrics
- **Learning Curves**: Per-model, per-CV-value learning curves with summary table and overfitting diagnosis
- **Dataset-Aware Output**: Graph titles and filenames include dataset name and target columns
- **Comprehensive Visualizations**: Confusion matrices, ROC curves, scatter plots, CV analysis charts
- **Dataset Agnostic**: Works with any CSV dataset (auto-detects delimiter) by configuring target columns

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML_CV

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- openpyxl

## Quick Start

```bash
# Run with default settings (CV=3,5 on all models)
python -X utf8 cv_experiment.py

# Test specific CV values
python -X utf8 cv_experiment.py --cv 3 5 10

# Test specific models only
python -X utf8 cv_experiment.py --models RandomForest SVC KNN

# Classification only
python -X utf8 cv_experiment.py --task classification --cv 3 5 10

# Regression only
python -X utf8 cv_experiment.py --task regression --cv 5 10

# Save output to file (capture both stdout and stderr)
python -X utf8 cv_experiment.py --cv 3 5 --task classification 2>&1 > .\output\results.txt
```

Or any combination of parameter control switches (--cv, --models, --task)

## Configuration

All settings are centralized in `config.py`:

### Dataset Configuration

```python
# Path to your dataset
DATA_PATH = DATA_DIR / 'ENB_data.csv'

# CSV delimiter (auto-detect or specify explicitly)
CSV_DELIMITER = 'auto'  # Options: 'auto', ',', ';', '\t'

# Columns to aggregate for target variable
# Single column: No aggregation needed, used directly as target
AGGREGATION_COLS = ['Temperature (C)']

# Multiple columns: Aggregation function combines them into single target
AGGREGATION_COLS = ['heating_load', 'cooling_load']

# Aggregation method (ignored for single column)
AGGREGATION_NAME = 'sum'  # Options: 'sum', 'mean', 'euclidean', 'manhattan', etc.

# Columns to exclude from features (will not be used for training)
EXCLUDE_COLS = ['id', 'Daily Summary']
```

### Data Preprocessing

```python
# Outlier removal using IQR method
REMOVE_OUTLIERS = True       # Set to False to skip
OUTLIER_IQR_MULTIPLIER = 1.5 # 1.5 = standard, 3.0 = extreme outliers only

# Polynomial features for linear regression models
# Only applied to: Ridge, Lasso, ElasticNet, LinearRegression
# Tree-based models (RF, GBT, XGBoost, LightGBM) use original features
USE_POLYNOMIAL_FEATURES = True
POLYNOMIAL_DEGREE = 2        # 2 = quadratic, 3 = cubic (slower)
```

### Model Settings

```python
RANDOM_STATE = 42        # Reproducibility seed
TEST_SIZE = 0.2          # Train/test split ratio
CV_FOLDS = 10            # Default cross-validation folds
N_JOBS = -1              # Use all CPU cores

# Hyperparameter tuning strategy
SEARCH_TYPE = 'grid'     # 'grid' (exhaustive) or 'random' (faster)
```

### Classification Settings

```python
N_CLASSES = 4                    # Number of classes (keep 3-5 for meaningful groups)
CLUSTERING_METHOD = 'kmeans'     # 'kmeans' or 'qcut' for label creation
```

### Learning Curves

```python
GENERATE_LEARNING_CURVES = True  # Set to False to skip (saves time)
```

### Categorical Encoding Settings

```python
# Columns with unique values <= threshold will be encoded
# Columns with unique values > threshold will be dropped
CATEGORICAL_ENCODING_THRESHOLD = 10   # Max unique values for encoding

# Encoding method: 'label' (LabelEncoder) or 'onehot' (OneHotEncoder)
CATEGORICAL_ENCODING_METHOD = 'label'
```

The preprocessing automatically handles:
- **Date columns**: Extracted to year, month, day, dayofweek, is_weekend features
- **Low-cardinality categoricals**: Encoded using Label or OneHot encoding
- **High-cardinality columns**: Dropped (too many unique values)
- **Outliers**: Removed using IQR method (configurable)

### Hyperparameter Overrides

Custom hyperparameter grids can be defined in `hyperparameters.py` without modifying the core tuning code. Overrides take priority over defaults; models not listed in the override file use the built-in grids.

```python
# hyperparameters.py - uncomment and modify to customize
CLF_PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
    },
}

REG_PARAM_GRIDS = {
    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    },
}
```

## Available Models

### Classification (8 models)
| Model | Description |
|-------|-------------|
| RandomForest | Ensemble of decision trees |
| DecisionTree | Single decision tree classifier |
| SVC | Support Vector Classifier |
| KNN | K-Nearest Neighbors |
| LogisticRegression | Logistic regression with regularization |
| NaiveBayes | Gaussian Naive Bayes |
| Bagging | Bootstrap aggregating ensemble |
| Stacking | Stacked ensemble of multiple models |

### Regression (12 models)
| Model | Description | Polynomial Features |
|-------|-------------|:---:|
| RandomForest | Random forest regressor | - |
| DecisionTree | Decision tree regressor | - |
| SVM | Support Vector Machine regressor | - |
| KNN | K-Nearest Neighbors regressor | - |
| Ridge | L2 regularized linear regression | Yes |
| Lasso | L1 regularized linear regression | Yes |
| ElasticNet | Combined L1/L2 regularization | Yes |
| LinearRegression | Ordinary least squares | Yes |
| GradientBoosting | Gradient boosted trees | - |
| XGBoost | Extreme gradient boosting | - |
| LightGBM | Light gradient boosting machine | - |
| Bagging | Bagging regressor | - |
| Stacking | Stacked ensemble regressor | - |

## Aggregation Functions

**Single target column**: No aggregation needed - the column is used directly as the regression target.

**Multiple target columns**: Combine them using various methods:

| Method | Formula | Use Case |
|--------|---------|----------|
| `sum` | col1 + col2 | Simple additive relationship |
| `mean` | (col1 + col2) / 2 | Average value |
| `euclidean` | sqrt(col1^2 + col2^2) | Distance-based |
| `manhattan` | \|col1\| + \|col2\| | Absolute distance |
| `geometric_mean` | sqrt(col1 * col2) | Multiplicative relationship |
| `harmonic_mean` | 2 / (1/col1 + 1/col2) | Rate averaging |
| `rms` | sqrt((col1^2 + col2^2) / 2) | Root mean square |
| `mahalanobis` | sqrt(x^T * Sigma^-1 * x) | Correlation-aware distance |
| `seuclidean` | Standardized Euclidean | Variance-normalized |
| `custom` | User-defined lambda | Custom formula in config |

### Custom Aggregation Example

```python
# In config.py
AGGREGATION_NAME = 'custom'
CUSTOM_AGGREGATION_FUNCTION = lambda col1, col2: 0.7 * col1 + 0.3 * col2
```

## Processing Pipeline

```
1. Load Data          - Auto-detect CSV delimiter (comma, semicolon, tab)
       |
2. Remove Outliers    - IQR method (configurable multiplier)
       |
3. Create Targets     - Aggregation + classification labels (qcut/kmeans)
       |
4. Feature Split      - Separate features (X) from targets (y)
       |                  Handle dates, categoricals, excluded columns
5. Train/Test Split   - Separate splits for classification and regression
       |
6. Feature Scaling    - StandardScaler (fit on train only)
       |
7. Polynomial (opt.)  - PolynomialFeatures for linear models only
       |
8. Model Training     - Hyperparameter tuning (Grid/Random search)
       |                  Per-model, per-CV-value learning curves
9. Evaluation         - Metrics collection, visualizations, reports
       |
10. Summary           - Learning curves summary table, CV analysis
```

## Project Structure

```
ML_CV/
├── config.py                 # Central configuration file
├── cv_experiment.py          # Main experiment runner
├── hyperparameters.py        # Optional hyperparameter overrides
├── core/
│   ├── preprocessing.py      # Data loading, outlier removal, preprocessing
│   ├── models.py             # Model training and learning curve generation
│   ├── tuning.py             # Hyperparameter tuning (with override support)
│   └── metrics.py            # Model evaluation metrics
├── utils/
│   └── aggregations.py       # Target aggregation functions
├── visualization/
│   ├── classification.py     # Confusion matrices, ROC curves
│   ├── regression.py         # Scatter plots, residuals
│   ├── analysis.py           # Feature distributions, learning curves
│   ├── cv_analysis.py        # CV comparison visualizations
│   └── learning_curves_summary.py  # Learning curves summary table
├── data/                     # Dataset directory
│   ├── ENB_data.csv
│   ├── winequality-red.csv
│   ├── energydata_complete.csv
│   └── ...
├── .vscode/
│   └── launch.json           # VSCode debug configuration
└── output/
    ├── graphs/               # Generated visualizations
    │   ├── classification/   # Confusion matrices, ROC curves
    │   ├── regression/       # Scatter plots
    │   ├── cv_analysis/      # CV comparison charts
    │   └── learning_curves/  # Per-model learning curves + summary
    └── reports/              # Excel reports and metrics
```

## Output

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score (Macro & Micro)
- AUC-ROC (micro/macro averaged)
- Confusion matrices
- ROC curves for each class

### Regression Metrics
- R2 Score
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### Learning Curves
- Per-model, per-CV-value learning curves (train vs CV score)
- Overfitting/underfitting diagnosis per model
- Summary table comparing all models with CV value, gap, and diagnosis

### CV Analysis Visualizations
- Bar charts comparing model performance across CV values
- Trend lines showing tuning time vs CV folds
- Heatmaps of performance metrics
- Best parameters summary

### Dataset-Aware Output
- Graph titles include dataset name and target column(s)
- Filenames include dataset suffix, CV value, and search type for easy organization

## Sample Datasets

The pipeline has been tested with:

| Dataset | Source | Target Columns | Delimiter |
|---------|--------|----------------|-----------|
| ENB_data.csv | [UCI](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) | heating_load, cooling_load | `,` |
| weather_data.csv | [Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset) | Temperature (C) | `,` |
| bike_sharing_hour.csv | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | cnt | `,` |
| winequality-red.csv | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) | quality | `;` |
| energydata_complete.csv | [UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | Appliances | `,` |
| superconductivty_train.csv | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | critical_temp | `,` |

Want to test your dataset, just assign it to `DATA_PATH` in config.py.

## Command Line Usage

```bash
# Full syntax
python -X utf8 cv_experiment.py [--cv CV_VALUES] [--models MODEL_NAMES] [--task TASK_TYPE]

# Arguments:
#   --cv       Space-separated CV fold values (default: 3 5)
#   --models   Space-separated model names (default: all)
#   --task     classification, regression, or both (default: both)

# Examples:
python -X utf8 cv_experiment.py --cv 3 5 10 15 --task classification
python -X utf8 cv_experiment.py --models XGBoost LightGBM --cv 5 10
python -X utf8 cv_experiment.py --models RandomForest --cv 3 5 10 --task both

# Save output with error capture
python -X utf8 cv_experiment.py --cv 3 5 --task regression 2>&1 > .\output\results.txt
```

### VSCode Debugging

A launch.json configuration is included for debugging with F5:

```json
{
    "name": "CV Experiment - Classification",
    "type": "debugpy",
    "request": "launch",
    "python": "${workspaceFolder}/venv/Scripts/python.exe",
    "program": "./cv_experiment.py",
    "console": "integratedTerminal",
    "pythonArgs": ["-X", "utf8"],
    "args": ["--cv", "3", "5", "10", "--task", "classification"]
}
```

## Tips for Best Results

1. **UTF-8 Encoding**: Always use `-X utf8` flag to avoid encoding issues:
   ```bash
   python -X utf8 cv_experiment.py > output.txt
   ```

2. **Capture Errors**: Use `2>&1` to capture both stdout and stderr:
   ```bash
   python -X utf8 cv_experiment.py 2>&1 > output.txt
   ```

3. **Fast Experimentation**: Use `SEARCH_TYPE = 'random'` for 20-30x faster tuning

4. **CV Selection**:
   - CV=3: Fast, good for initial testing
   - CV=5: Balanced (recommended)
   - CV=10: More reliable, slower

5. **N_CLASSES**: Keep between 3-5 regardless of unique values in target column. Higher values create classes with too few samples

6. **Outlier Removal**: Set `REMOVE_OUTLIERS = True` with `OUTLIER_IQR_MULTIPLIER = 1.5` for standard cleaning, `3.0` for extreme outliers only

7. **Polynomial Features**: Enable for better linear model performance on non-linear data. Degree=2 is recommended (degree=3 is much slower)

8. **Memory Issues**: Reduce `N_JOBS` if running out of memory

9. **Overfitting in Learning Curves**: A train-CV gap of 0.10-0.15 is normal for tree-based models. Focus on the CV score, not the gap

10. **Wine Quality Datasets**: These use semicolons (`;`) as delimiters. Set `CSV_DELIMITER = 'auto'` or `';'`

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and pull requests for improvements.
