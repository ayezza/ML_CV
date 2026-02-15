"""
Configuration file for ML Pipeline
All settings and constants are defined here
"""

from pathlib import Path

class Config:
    """Configuration class for the ML pipeline"""

    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    OUTPUT_DIR = BASE_DIR / 'output'

    # Input datasets paths already cleaned and optimized, that can be used for testing (you can add yours)
    # /////////////////////////////////////////////////////////////////////////////////////
    # ENB_data.csv: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset, 
    # bike_sharing_hour.csv:  https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset, 
    # weather_data.csv: https://www.kaggle.com/datasets/muthuj7/weather-dataset,
    # winequality-red.csv & winequality-white.csv: https://archive.ics.uci.edu/dataset/186/wine+quality
    # energydata_complete.csv:  https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    # superconductivty_train.csv & superconductivty_unique_m.csv: https://archive.ics.uci.edu/dataset/464/superconductivty+data
    DATA_PATH = DATA_DIR / 'ENB_data.csv'

    # CSV delimiter setting
    # Options: 'auto' (auto-detect), ',' (comma), ';' (semicolon), '\t' (tab), etc.
    # Note: UCI wine quality datasets use semicolons as delimiters
    CSV_DELIMITER = 'auto'  # Auto-detect delimiter (or use ',' or ';' explicitly)

    # Outlier removal settings (IQR method)
    REMOVE_OUTLIERS = True  # Set to False to skip outlier removal depending on dataset and model choice
    # iqr = q3 - q1, lower_bound = q1 - 1.5 * iqr, upper_bound = q3 + 1.5 * iqr
    OUTLIER_IQR_MULTIPLIER = 1.5  # 1.5 = standard (moderate), 3.0 = extreme outliers only

    # Polynomial features settings (for linear regression models only)
    # Only applied to: Ridge, Lasso, ElasticNet, LinearRegression
    # Tree-based models (RF, GBT, XGBoost, LightGBM) use original features
    USE_POLYNOMIAL_FEATURES = False  # Set to False to skip polynomial transformation
    POLYNOMIAL_DEGREE = 2  # Degree of polynomial features (2 = quadratic, 3 = cubic)

    # Aggregation settings (replace the value for testing)
    # Columns to aggregate for target variable if multiple columns are specified, 
    # they will be combined using the function defined in AGGREGATION_NAME
    AGGREGATION_COLS = ['heating_load', 'cooling_load']  
    # these columns must exist in the dataset

    # Columns to exclude from features (will not be used for training)
    EXCLUDE_COLS = ['heating_load', 'cooling_load']  


    # Aggregation method only if multiple columns are specified in AGGREGATION_COLS
    AGGREGATION_NAME = 'sum'  # Options: 'sum', 'mean', 'geometric_mean', 'manhattan', 'euclidean', 'rms', 'weighted', 'weighted_70_30', 'power_mean_3', 'chebyshev', 'minkowski', 'seuclidean', 'mahalanobis', 'custom'

    # Custom aggregation function (used when AGGREGATION_NAME = 'custom')
    CUSTOM_AGGREGATION_FUNCTION = lambda h, c: .920 * h + 4.064  # c not used - derived from h

    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 10  # Cross-validation folds for hyperparameter tuning
                  # Options: 3 (fast), 5 (balanced - recommended), 10 (reliable but slower)
    N_JOBS = -1  # Use all CPU cores

    # Hyperparameter tuning strategy
    SEARCH_TYPE = 'random'  # Options: 'grid' (exhaustive but slow) or 'random' (fast, 20-30x faster)
                            # 'grid': Tests all parameter combinations (can take hours with large grids)
                            # 'random': Randomly samples fixed combinations (typically within 1-2% of optimal)
                            # Recommendation: Use 'random' for testing/experimentation, 'grid' for final runs

    # Classification settings
    # N_CLASSES: Number of bins for converting continuous targets into classification labels
    # Keep this small (3-5) regardless of unique values in the target column
    # 92 unique values does NOT mean 92 classes - that creates classes with 1-2 samples each
    N_CLASSES = 'auto'  # Recommended: 3-5 for meaningful classification groups

    # Categorical encoding settings
    # Columns with unique values <= threshold will be encoded (label or onehot)
    # Columns with unique values > threshold will be dropped (too many categories)
    CATEGORICAL_ENCODING_THRESHOLD = 100  # Max unique values for encoding
    CATEGORICAL_ENCODING_METHOD = 'label'  # Options: 'label' (LabelEncoder) or 'onehot' (OneHotEncoder)

    # Clustering method for creating classification labels
    # Options: 'qcut' (default - predefined bins) or 'kmeans' (natural clusters)
    # 'qcut': Uses pd.qcut to create equal-frequency bins based on aggregated values
    # 'kmeans': Uses KMeans clustering to find natural groupings in (heating, cooling) space
    # Note: KMeans often produces better class separation and may improve classification accuracy
    CLUSTERING_METHOD = 'kmeans'  # Options: 'qcut', 'kmeans'

    # Learning curve settings
    GENERATE_LEARNING_CURVES = True  # Set to False to skip learning curve generation (saves time)
    # Training set size fractions for learning curves
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, .45, 0.5, .55, 0.6, .65,  0.7, .75,  0.8, .85, 0.9, .95, 1.0]  
    # or LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)

    GENERATE_PREDS = False  # Whether to save model predictions to CSV files



    # Output subdirectories
    GRAPHS_DIR = OUTPUT_DIR / 'graphs'
    DATA_OUTPUT_DIR = OUTPUT_DIR / 'data'
    REPORTS_DIR = OUTPUT_DIR / 'reports'

    # Classification output directories
    CLF_GRAPHS_DIR = GRAPHS_DIR / 'classification'
    CLF_BASE_DIR = CLF_GRAPHS_DIR / 'base'
    CLF_TUNED_DIR = CLF_GRAPHS_DIR / 'tuned'

    # Classification subdirectories
    CLF_BASE_CONF_MATRIX_DIR = CLF_BASE_DIR / 'confusion_matrices'
    CLF_BASE_ROC_DIR = CLF_BASE_DIR / 'roc_curves'
    CLF_BASE_BAR_DIR = CLF_BASE_DIR / 'bar_charts'
    CLF_BASE_PROB_DIR = CLF_BASE_DIR / 'probability_matrices'

    CLF_TUNED_CONF_MATRIX_DIR = CLF_TUNED_DIR / 'confusion_matrices'
    CLF_TUNED_ROC_DIR = CLF_TUNED_DIR / 'roc_curves'
    CLF_TUNED_BAR_DIR = CLF_TUNED_DIR / 'bar_charts'

    # Regression output directories
    REG_GRAPHS_DIR = GRAPHS_DIR / 'regression'
    REG_BASE_DIR = REG_GRAPHS_DIR / 'base'
    REG_TUNED_DIR = REG_GRAPHS_DIR / 'tuned'

    # Regression subdirectories
    REG_BASE_SCATTER_DIR = REG_BASE_DIR / 'scatter_plots'
    REG_TUNED_SCATTER_DIR = REG_TUNED_DIR / 'scatter_plots'

    # Learning curves directories
    LEARNING_CURVES_DIR = GRAPHS_DIR / 'learning_curves'
    CLF_LEARNING_CURVES_DIR = LEARNING_CURVES_DIR / 'classification'
    REG_LEARNING_CURVES_DIR = LEARNING_CURVES_DIR / 'regression'

    # Data output directories
    PREPROCESSED_DIR = DATA_OUTPUT_DIR / 'preprocessed'
    PREDICTIONS_DIR = DATA_OUTPUT_DIR / 'predictions'
    PROB_MATRICES_DIR = DATA_OUTPUT_DIR / 'probability_matrices'

    # Reports directories
    METRICS_DIR = REPORTS_DIR / 'metrics'
    CLF_REPORTS_DIR = REPORTS_DIR / 'classification_reports'

    # Predictions directories
    CLF_PREDICTIONS_DIR = REPORTS_DIR / 'classification_pred'
    REG_PREDICTIONS_DIR = REPORTS_DIR / 'regression_pred'

    


    # Plot settings
    DPI = 300 # Dots Per Inch
    FIGURE_SIZE_SMALL = (8, 6)
    FIGURE_SIZE_MEDIUM = (10, 8)
    FIGURE_SIZE_LARGE = (12, 10)

    # Matplotlib backend
    MATPLOTLIB_BACKEND = 'Agg'  # For saving to file (no-GUI display)

    @classmethod
    def create_directories(cls):
        """Create all output directories if they don't exist"""
        directories = [
            # Classification base
            cls.CLF_BASE_CONF_MATRIX_DIR,
            cls.CLF_BASE_ROC_DIR,
            cls.CLF_BASE_BAR_DIR,
            cls.CLF_BASE_PROB_DIR,
            # Classification tuned
            cls.CLF_TUNED_CONF_MATRIX_DIR,
            cls.CLF_TUNED_ROC_DIR,
            cls.CLF_TUNED_BAR_DIR,
            # Regression base
            cls.REG_BASE_SCATTER_DIR,
            # Regression tuned
            cls.REG_TUNED_SCATTER_DIR,
            # Learning curves
            cls.CLF_LEARNING_CURVES_DIR,
            cls.REG_LEARNING_CURVES_DIR,
            # Data outputs
            cls.PREPROCESSED_DIR,
            cls.PREDICTIONS_DIR,
            cls.PROB_MATRICES_DIR,
            # Reports
            cls.METRICS_DIR,
            cls.CLF_REPORTS_DIR,
            # Predictions
            cls.CLF_PREDICTIONS_DIR,
            cls.REG_PREDICTIONS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✓ All output directories created")

    @classmethod
    def get_output_path(cls, category, subcategory, is_tuned=False):
        """
        Get output path for a specific visualization type

        Args:
            category: 'classification' or 'regression'
            subcategory: 'confusion_matrix', 'roc_curve', 'bar_chart', 'scatter_plot', etc.
            is_tuned: Whether this is for a tuned model

        Returns:
            Path object for the output directory
        """
        if category == 'classification':
            base_dir = cls.CLF_TUNED_DIR if is_tuned else cls.CLF_BASE_DIR
            mapping = {
                'confusion_matrix': 'confusion_matrices',
                'roc_curve': 'roc_curves',
                'bar_chart': 'bar_charts',
                'probability_matrix': 'probability_matrices'
            }
        elif category == 'regression':
            base_dir = cls.REG_TUNED_DIR if is_tuned else cls.REG_BASE_DIR
            mapping = {
                'scatter_plot': 'scatter_plots'
            }
        else:
            raise ValueError(f"Unknown category: {category}")

        if subcategory not in mapping:
            raise ValueError(f"Unknown subcategory: {subcategory}")

        return base_dir / mapping[subcategory]

    @classmethod
    def get_dataset_name(cls):
        """Get the dataset filename without extension"""
        return cls.DATA_PATH.stem

    @classmethod
    def get_target_cols_str(cls):
        """Get target columns as a readable string for titles"""
        return ', '.join(cls.AGGREGATION_COLS)

    @classmethod
    def get_filename_suffix(cls):
        """Get a sanitized suffix for filenames (dataset + target cols)"""
        dataset = cls.DATA_PATH.stem
        # Sanitize target cols for filename (replace spaces and special chars)
        target_str = '_'.join(cls.AGGREGATION_COLS)
        target_str = target_str.replace(' ', '_').replace('(', '').replace(')', '')
        return f"{dataset}_{target_str}"

    @classmethod
    def get_dataset_info(cls):
        """Get dataset info dict for visualization functions"""
        return {
            'name': cls.get_dataset_name(),
            'target_cols': cls.get_target_cols_str(),
            'filename_suffix': cls.get_filename_suffix()
        }

    @classmethod
    def summary(cls):
        """Print configuration summary"""
        print("\n" + "="*70)
        print(" "*20 + "CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Data Path:          {cls.DATA_PATH}")
        print(f"Output Directory:   {cls.OUTPUT_DIR}")
        print(f"DATA DIRECTORY:     {cls.DATA_DIR}")
        print(f"GRAPHS DIRECTORY:   {cls.GRAPHS_DIR}")
        print(f"REPORTS DIRECTORY:  {cls.REPORTS_DIR}")
        print(f"DATA OUTPUT DIR:    {cls.DATA_OUTPUT_DIR}")
        print(f"CLASSIFICATION GRAPHS DIRECTORY:   {cls.CLF_GRAPHS_DIR}")
        print(f"REGRESSION GRAPHS DIRECTORY:       {cls.REG_GRAPHS_DIR}")
        print(f"LEARNING CURVES DIRECTORY:         {cls.LEARNING_CURVES_DIR}")
        print(f"PREPROCESSED DATA DIRECTORY:       {cls.PREPROCESSED_DIR}")
        print(f"PREDICTIONS DIRECTORY:             {cls.PREDICTIONS_DIR}")
        print(f"METRICS REPORTS DIRECTORY:         {cls.METRICS_DIR}")
        print(f"CLASSIFICATION REPORTS DIRECTORY:  {cls.CLF_REPORTS_DIR}")
        print(f"CLASSIFICATION PREDICTIONS DIRECTORY: {cls.CLF_PREDICTIONS_DIR}")
        print(f"REGRESSION PREDICTIONS DIRECTORY:      {cls.REG_PREDICTIONS_DIR}")
        print(f"Dataset Name:       {cls.get_dataset_name()}")
        print(f"CSV Delimiter:      {cls.CSV_DELIMITER}")
        print(f"Remove Outliers:    {cls.REMOVE_OUTLIERS} (IQR multiplier: {cls.OUTLIER_IQR_MULTIPLIER})")
        print(f"Polynomial Features: {cls.USE_POLYNOMIAL_FEATURES} (degree: {cls.POLYNOMIAL_DEGREE})")
        print(f"Exclude Columns:    {cls.EXCLUDE_COLS}")
        print(f"Target Columns:     {cls.get_target_cols_str()}")
        print(f"AGGREGATION COLUMNS:  {cls.AGGREGATION_COLS}")
        print(f"Categorical Encoding Threshold: {cls.CATEGORICAL_ENCODING_THRESHOLD}")
        print(f"Categorical Encoding Method:    {cls.CATEGORICAL_ENCODING_METHOD}")
        print(f"Aggregation:        {cls.AGGREGATION_NAME}")
        print(f"Random State:       {cls.RANDOM_STATE}")
        print(f"Test Size:          {cls.TEST_SIZE}")
        print(f"CV Folds:           {cls.CV_FOLDS}")
        print(f"N Classes:          {cls.N_CLASSES}")
        print(f"DPI:                {cls.DPI}")
        print(f"Matplotlib Backend: {cls.MATPLOTLIB_BACKEND}")
        print(f"Generate Learning Curves: {cls.GENERATE_LEARNING_CURVES}")
        print(f"Learning Curve Train Sizes: {cls.LEARNING_CURVE_TRAIN_SIZES}")
        print(f"Search Type:        {cls.SEARCH_TYPE}")
        print(f"Clustering Method:  {cls.CLUSTERING_METHOD}")
        print(f"Generate Predictions:      {cls.GENERATE_PREDS}")
        print(f"")
        print("="*70 + "\n")
