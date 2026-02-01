# ML Cross-Validation Pipeline

A comprehensive machine learning pipeline for testing and comparing classification and regression models with cross-validation experiments. This project enables systematic evaluation of different CV fold values, hyperparameter tuning strategies, and model performance across multiple algorithms.

## Features

- **Multi-Model Comparison**: Test 8 classification and 12 regression models simultaneously
- **Cross-Validation Experiments**: Compare CV=3, 5, 10, 15 to find optimal settings for your data
- **Hyperparameter Tuning**: Support for GridSearchCV and RandomizedSearchCV (more time efficient)
- **Automatic Preprocessing**: Handles date columns, non-numeric data, and feature scaling for fast convergence
- **Multiple Aggregation Functions**: 17+ methods to combine target columns (sum, euclidean, mahalanobis, etc.)
- **Auto Class Detection**: Automatically finds optimal number of classes using clustering metrics (but you can fix your own classes)
- **Comprehensive Visualizations**: Confusion matrices, ROC curves, scatter plots, CV analysis charts
- **Dataset Agnostic**: Works with any CSV dataset by configuring target columns

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
python cv_experiment.py

# Test specific CV values
python cv_experiment.py --cv 3 5 10

# Test specific models only
python cv_experiment.py --models RandomForest SVC KNN

# Classification only
python cv_experiment.py --task classification --cv 3 5 10

# Regression only
python cv_experiment.py --task regression --cv 5 10
```

Or any combination of parameter control switches (--cv, --models, --task)


## Configuration

All settings are centralized in `config.py`:

### Dataset Configuration

```python
# Path to your dataset
DATA_PATH = DATA_DIR / 'ENB_data.csv'

# Columns to aggregate for target variable
AGGREGATION_COLS = ['heating_load', 'cooling_load']

# Aggregation method
AGGREGATION_NAME = 'sum'  # Options: 'sum', 'mean', 'euclidean', 'manhattan', etc.
```

### Model Settings

```python
RANDOM_STATE = 42        # Reproducibility seed
TEST_SIZE = 0.2          # Train/test split ratio
CV_FOLDS = 10            # Default cross-validation folds
N_JOBS = -1              # Use all CPU cores

# Hyperparameter tuning
SEARCH_TYPE = 'grid'     # 'grid' (exhaustive) or 'random' (faster)
```

### Classification Settings

```python
N_CLASSES = 'auto'               # Auto-detect optimal number of classes
CLUSTERING_METHOD = 'kmeans'     # 'kmeans' or 'qcut' for label creation
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
| Model | Description |
|-------|-------------|
| RandomForest | Random forest regressor |
| DecisionTree | Decision tree regressor |
| SVM | Support Vector Machine regressor |
| KNN | K-Nearest Neighbors regressor |
| Ridge | L2 regularized linear regression |
| Lasso | L1 regularized linear regression |
| ElasticNet | Combined L1/L2 regularization |
| GradientBoosting | Gradient boosted trees |
| XGBoost | Extreme gradient boosting |
| LightGBM | Light gradient boosting machine |
| Bagging | Bagging regressor |
| Stacking | Stacked ensemble regressor |

## Aggregation Functions

Combine multiple target columns using various methods:

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

## Project Structure

```
ML_CV/
├── config.py                 # Central configuration file
├── cv_experiment.py          # Main experiment runner
├── core/
│   ├── preprocessing.py      # Data loading and preprocessing
│   ├── tuning.py             # Hyperparameter tuning
│   └── metrics.py            # Model evaluation metrics
├── utils/
│   └── aggregations.py       # Target aggregation functions
├── visualization/
│   ├── classification.py     # Confusion matrices, ROC curves
│   ├── regression.py         # Scatter plots, residuals
│   ├── analysis.py           # Feature distributions
│   └── cv_analysis.py        # CV comparison visualizations
├── data/                     # Dataset directory
│   ├── ENB_data.csv          # Energy efficiency dataset
│   ├── bike_sharing/         # Bike sharing dataset
│   └── wine_quality/         # Wine quality dataset
└── output/
    ├── graphs/               # Generated visualizations
    │   ├── classification/
    │   ├── regression/
    │   └── cv_analysis/
    └── reports/              # Excel reports and metrics
```

## Output Examples

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (micro/macro averaged)
- Confusion matrices
- ROC curves for each class

### Regression Metrics
- R2 Score
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### CV Analysis Visualizations
- Bar charts comparing model performance across CV values
- Trend lines showing tuning time vs CV folds
- Heatmaps of performance metrics
- Best parameters summary

## Sample Datasets

The pipeline has been tested with:

| Dataset | Source | Target Columns |
|---------|--------|----------------|
| ENB_data.csv | [UCI](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) | heating_load, cooling_load |
| bike_sharing_hour.csv | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | cnt (count) |
| winequality-red.csv | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) | quality |
| energydata_complete.csv | [UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | Appliances |
| superconductivty_train.csv | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) | critical_temp |

Want to test your dataset, just affect it to `DATA_PATH` global variable. 

## Command Line Usage

```bash
# Full syntax
python cv_experiment.py [--cv CV_VALUES] [--models MODEL_NAMES] [--task TASK_TYPE]

# Arguments:
#   --cv       Space-separated CV fold values (default: 3 5)
#   --models   Space-separated model names (default: all)
#   --task     classification, regression, or both (default: both)

# Examples:
python cv_experiment.py --cv 3 5 10 15 --task classification
python cv_experiment.py --models XGBoost LightGBM --cv 5 10
python cv_experiment.py --models RandomForest --cv 3 5 10 --task both
```

## Tips for Best Results

1. **UTF-8 Encoding**: If redirecting output to file, use:
   ```bash
   python -X utf8 cv_experiment.py > output.txt
   ```

2. **Fast Experimentation**: Use `SEARCH_TYPE = 'random'` for 20-30x faster tuning

3. **CV Selection**:
   - CV=3: Fast, good for initial testing
   - CV=5: Balanced (recommended)
   - CV=10: More reliable, slower

4. **Memory Issues**: Reduce `N_JOBS` if running out of memory

5. **Convergence Warnings**: Data is automatically scaled, but if warnings persist, check your data for outliers and make the appropriate data cleaning

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and pull requests for improvements.
