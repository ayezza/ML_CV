"""
Data Preprocessing Module

This module handles data loading, preprocessing, and feature engineering.
"""
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.aggregations import get_aggregation_function
# Metrics for clustering evaluation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from config import Config


class DataPreprocessor:
    """
    Handles data loading and preprocessing operations

    This class provides methods to load data, compute correlations,
    create target variables, and prepare data for modeling.
    """

    def __init__(self, data_path, aggregation_cols=None, aggregation_name='sum', aggregated_var_name='charges_sum', class_var_name='charges_classes'):
        """
        Initialize the preprocessor

        Args:
            data_path: Path to the input CSV file
            aggregation_cols: List of columns to aggregate
            aggregation_name: Name of aggregation function to use
            aggregated_var_name: Name of the aggregated target variable
            class_var_name: Name of the classification target variable
        """
        self.data_path = Path(data_path)
        self.aggregation_cols = aggregation_cols
        self.aggregation_name = aggregation_name
        self.aggregated_var_name = aggregated_var_name
        self.class_var_name = class_var_name
        self.df = None
        self.df_corr = None
        self.target_classes = {}

    def load_data(self):
        """
        Load dataset from CSV file

        Returns:
            DataFrame: The loaded dataframe

        Raises:
            FileNotFoundError: If the data file does not exist
            PermissionError: If there are permission issues accessing the file
            pd.errors.EmptyDataError: If the CSV file is empty
            pd.errors.ParserError: If the CSV file is malformed
            ValueError: If the loaded data is invalid

        Example:
            >>> preprocessor = DataPreprocessor('./ENB_data.csv')
            >>> df = preprocessor.load_data()
        """
        print("="*80)
        print(" "*25 + "LOADING DATASET")
        print("="*80)

        # Check if data_path exists
        if not self.data_path.exists():
            error_msg = f"Data file not found: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            print(f"   Current working directory: {Path.cwd()}")
            print(f"   Please ensure the file exists at the specified path.")
            raise FileNotFoundError(error_msg)

        # Check if it's a file (not a directory)
        if not self.data_path.is_file():
            error_msg = f"Path exists but is not a file: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            raise ValueError(error_msg)

        # Check file extension
        if self.data_path.suffix.lower() not in ['.csv', '.txt']:
            print(f"\nWARNING: File extension is '{self.data_path.suffix}', expected '.csv' or '.txt'")

        try:
            # Attempt to read the CSV file
            print(f"\nLoading file: {self.data_path.name}")
            print(f"   Full path: {self.data_path.absolute()}")

            # Get delimiter from config (supports 'auto' for auto-detection)
            delimiter = getattr(Config, 'CSV_DELIMITER', ',')
            if delimiter == 'auto':
                # Auto-detect delimiter by reading first line
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                # Check common delimiters
                if ';' in first_line and ',' not in first_line:
                    delimiter = ';'
                elif '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                else:
                    delimiter = ','
                print(f"   Auto-detected delimiter: '{delimiter}'")
            else:
                print(f"   Using delimiter: '{delimiter}'")

            self.df = pd.read_csv(self.data_path, delimiter=delimiter)

            # Validate loaded data
            if self.df.empty:
                raise ValueError("Loaded dataframe is empty (no rows)")

            if len(self.df.columns) == 0:
                raise ValueError("Loaded dataframe has no columns")

            print(f"\n✓ Dataset loaded successfully")
            print(f"  Shape: {self.df.shape}")
            print(f"  Columns: {self.df.columns.tolist()}")

            print("\nFirst 5 rows:")
            print(self.df.head())

            print("\nDataset info:")
            self.df.info()

            print("\nDataset description:")
            print(self.df.describe())
            print("="*80 + "\n")

            return self.df

        except FileNotFoundError as e:
            # Re-raise FileNotFoundError (already handled above, but just in case)
            raise

        except PermissionError as e:
            error_msg = f"Permission denied when accessing file: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            print(f"   Make sure you have read permissions for this file.")
            raise PermissionError(error_msg) from e

        except pd.errors.EmptyDataError as e:
            error_msg = f"CSV file is empty: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            raise pd.errors.EmptyDataError(error_msg) from e

        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file (malformed data): {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            print(f"   Details: {str(e)}")
            print(f"   The CSV file may be corrupted or have incorrect formatting.")
            raise pd.errors.ParserError(error_msg) from e

        except UnicodeDecodeError as e:
            error_msg = f"Encoding error when reading file: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            print(f"   Try specifying encoding explicitly (e.g., encoding='utf-8' or 'latin-1')")
            raise ValueError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error loading data: {self.data_path.absolute()}"
            print(f"\nERROR: {error_msg}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Details: {str(e)}")
            raise RuntimeError(error_msg) from e

    def compute_correlations(self, method='pearson'):
        """
        Compute correlation matrix

        Args:
            method: Correlation method ('pearson', 'kendall', 'spearman')

        Returns:
            DataFrame: Correlation matrix

        Example:
            >>> preprocessor.compute_correlations()
        """
        print("="*80)
        print(" "*25 + "CORRELATION ANALYSIS")
        print("="*80)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df_corr = self.df.corr(method)

        print("\nCorrelation matrix:")
        print(self.df_corr)

        print("\n" + "-"*80)
        print("KEY CORRELATIONS WITH TARGET VARIABLES")
        print("-"*80)

        # Analyze correlations with heating and cooling loads
        target_vars = self.aggregation_cols
        if target_vars is None:
            print("No aggregation columns specified for correlation analysis.")
            return self.df_corr
        else:
            for target in target_vars:
                if target in self.df_corr.columns:
                    correlations = self.df_corr[target].drop(target)
                    sorted_correlations = correlations.abs().sort_values(ascending=False)
                    print(f"\nTop correlated features with {target}:")
                    print(sorted_correlations.head())

            # print("\n" + "-"*80)
            # print("INSIGHTS:")
            # print("-"*80)
            # print(f"1. Heating and cooling loads are highly correlated (~0.98)")
            # print("2. Both are strongly correlated with:")
            # print("   - overall_height (positive: ~0.89-0.90)")
            # print("   - roof_area (negative: ~-0.86)")
            # print("3. Moderately correlated with wall_area and glazing_area")
            # print("="*80 + "\n")

        return self.df_corr

    def find_optimal_n_classes(self, k_range=(2, 10), method='silhouette'):
        """
        Find the optimal number of classes/clusters using clustering metrics.

        Args:
            k_range: Tuple (min_k, max_k) - range of k values to test (default: 2 to 10)
            method: Method to determine optimal k:
                    'silhouette' - Maximize silhouette score (default, recommended)
                    'calinski' - Maximize Calinski-Harabasz index
                    'davies_bouldin' - Minimize Davies-Bouldin index
                    'elbow' - Use elbow method (inertia)

        Returns:
            dict: {
                'optimal_k': int - Best number of clusters,
                'scores': dict - All scores for each k,
                'method': str - Method used
            }

        Example:
            >>> result = preprocessor.find_optimal_n_classes(k_range=(2, 8))
            >>> optimal_k = result['optimal_k']
            >>> preprocessor.create_target_variables(n_classes=optimal_k, clustering_method='kmeans')
        """
        print("="*80)
        print(" "*15 + "FINDING OPTIMAL NUMBER OF CLASSES")
        print("="*80)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Prepare data for clustering
        X = self.df[self.aggregation_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        min_k, max_k = k_range
        k_values = range(min_k, max_k + 1)

        # Store metrics for each k
        silhouette_scores = {}
        calinski_scores = {}
        davies_bouldin_scores = {}
        inertias = {}

        print(f"\nTesting k values from {min_k} to {max_k}...")
        print(f"Method: {method.upper()}\n")
        print(f"{'k':<4} {'Silhouette':<12} {'Calinski-H':<12} {'Davies-B':<12} {'Inertia':<12}")
        print("-" * 56)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Calculate all metrics
            sil_score = silhouette_score(X_scaled, labels)
            cal_score = calinski_harabasz_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            inertia = kmeans.inertia_

            silhouette_scores[k] = sil_score
            calinski_scores[k] = cal_score
            davies_bouldin_scores[k] = db_score
            inertias[k] = inertia

            print(f"{k:<4} {sil_score:<12.4f} {cal_score:<12.2f} {db_score:<12.4f} {inertia:<12.2f}")

        # Determine optimal k based on method
        if method == 'silhouette':
            optimal_k = max(silhouette_scores, key=silhouette_scores.get)
            best_score = silhouette_scores[optimal_k]
            print(f"\n✓ Optimal k={optimal_k} (highest silhouette score: {best_score:.4f})")

        elif method == 'calinski':
            optimal_k = max(calinski_scores, key=calinski_scores.get)
            best_score = calinski_scores[optimal_k]
            print(f"\n✓ Optimal k={optimal_k} (highest Calinski-Harabasz: {best_score:.2f})")

        elif method == 'davies_bouldin':
            optimal_k = min(davies_bouldin_scores, key=davies_bouldin_scores.get)
            best_score = davies_bouldin_scores[optimal_k]
            print(f"\n✓ Optimal k={optimal_k} (lowest Davies-Bouldin: {best_score:.4f})")

        elif method == 'elbow':
            # Find elbow using second derivative (rate of change)
            inertia_list = [inertias[k] for k in k_values]
            # Calculate rate of change
            diffs = np.diff(inertia_list)
            # Find where the rate of change slows down the most
            elbow_idx = np.argmin(np.diff(diffs)) + 1
            optimal_k = list(k_values)[elbow_idx]
            print(f"\n✓ Optimal k={optimal_k} (elbow point in inertia curve)")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'silhouette', 'calinski', 'davies_bouldin', or 'elbow'")

        print("="*80 + "\n")

        return {
            'optimal_k': optimal_k,
            'scores': {
                'silhouette': silhouette_scores,
                'calinski': calinski_scores,
                'davies_bouldin': davies_bouldin_scores,
                'inertia': inertias
            },
            'method': method
        }

    def create_target_variables(self, n_classes=4, clustering_method='qcut', auto_k_method='silhouette', create_classes=True):
        """
        Create aggregated target variable and optionally classification classes

        Args:
            n_classes: Number of classes for classification (default: 4)
                       Use 'auto' to automatically find optimal number of classes
            clustering_method: Method to create classes - 'qcut' or 'kmeans' (default: 'qcut')
                'qcut': Quantile-based binning on aggregated values
                'kmeans': KMeans clustering on target_vars space
            auto_k_method: Method for finding optimal k when n_classes='auto'
                           Options: 'silhouette', 'calinski', 'davies_bouldin', 'elbow'
            create_classes: Whether to create classification classes (default: True)
                           Set to False when only regression is needed (saves time)

        Returns:
            tuple: (df, target_classes_dict) - Updated dataframe and dict of class DataFrames

        Example:
            >>> df, classes = preprocessor.create_target_variables(n_classes=4, clustering_method='kmeans')
            >>> # Or find optimal k automatically:
            >>> df, classes = preprocessor.create_target_variables(n_classes='auto', clustering_method='kmeans')
            >>> # For regression only (skip class creation):
            >>> df, classes = preprocessor.create_target_variables(create_classes=False)
        """
        print("="*80)
        print(" "*20 + "CREATING TARGET VARIABLES")
        print("="*80)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Check if single column (no aggregation needed) or multiple columns
        n_agg_cols = len(self.aggregation_cols) if self.aggregation_cols else 0

        if n_agg_cols == 0:
            raise ValueError("No aggregation columns specified. Set AGGREGATION_COLS in config.")

        # Handle 'auto' n_classes - find optimal k (only if creating classes)
        if create_classes and n_classes == 'auto':
            result = self.find_optimal_n_classes(k_range=(2, 10), method=auto_k_method)
            n_classes = result['optimal_k']
            print(f"Using automatically determined n_classes={n_classes}")

        if n_agg_cols == 1:
            # Single column - no aggregation needed, use column directly
            target_col = self.aggregation_cols[0]
            print(f"\nSingle target column: '{target_col}' (no aggregation needed)")
            self.df[self.aggregated_var_name] = self.df[target_col]
            print(f"\nFirst 5 rows of target '{target_col}':")
            print(self.df[[target_col]].head())
        else:
            # Multiple columns - apply aggregation function
            aggregation_func = get_aggregation_function(self.aggregation_name)
            print(f"\nUsing aggregation: {self.aggregation_name}")
            print(f"Formula: {self._get_aggregation_formula()}")

            # Create aggregated target
            self.df[self.aggregated_var_name] = aggregation_func(
                self.df[self.aggregation_cols[0]],
                self.df[self.aggregation_cols[1]]
            )

            print("\nFirst 5 rows with aggregated target:")
            print(self.df[[self.aggregation_cols[0], self.aggregation_cols[1], self.aggregated_var_name]].head())

        # Create classification classes only if needed
        if create_classes:
            # Create classification classes based on specified method
            print(f"\nClustering method: {clustering_method.upper()}")

            if clustering_method == 'kmeans':
                # Use KMeans clustering on target_vars space
                print(f"Using KMeans clustering to find {n_classes} natural clusters...")

                # Prepare data for clustering (handle single or multiple columns)
                if n_agg_cols == 1:
                    X = self.df[[self.aggregation_cols[0]]].values
                else:
                    X = self.df[self.aggregation_cols].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)

                # Assign cluster labels as classes
                self.df[self.class_var_name] = cluster_labels
                # Calculate cluster quality metrics
                silhouette = silhouette_score(X_scaled, cluster_labels)
                calinski = calinski_harabasz_score(X_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

                print(f"✓ Created {n_classes} classes using KMeans clustering")
                print(f"  Cluster quality metrics:")
                print(f"    Silhouette Score:       {silhouette:.4f}  (higher is better, -1 to 1)")
                print(f"    Calinski-Harabasz:      {calinski:.2f}   (higher is better)")
                print(f"    Davies-Bouldin:         {davies_bouldin:.4f}  (lower is better)")

                # Get cluster centers (in original scale for interpretability)
                centers_scaled = kmeans.cluster_centers_
                centers = scaler.inverse_transform(centers_scaled)

                print(f"\n  Cluster centers {self.aggregation_cols}:")
                if n_agg_cols == 1:
                    for i, center in enumerate(centers):
                        print(f"    Cluster {i}: ({center[0]:.2f})")
                else:
                    for i, center in enumerate(centers):
                        print(f"    Cluster {i}: ({', '.join(f'{v:.2f}' for v in center)})")

            else:  # pandas qcut (default)
                # Use quantile-based binning on aggregated values
                self.df[self.class_var_name] = pd.qcut(
                    self.df[self.aggregated_var_name],
                    q=n_classes,
                    labels=list(range(n_classes)),
                    duplicates='drop'
                )

                print(f"✓ Created {n_classes} classes using quantile-based binning (qcut)")

            # Split data by class for analysis
            self.target_classes = {}
            for i in range(n_classes):
                class_df = self.df[self.df[self.class_var_name] == i].sort_values(
                    by=self.aggregated_var_name, ascending=True
                )
                self.target_classes[i] = class_df
                print(f"\nClass {i}: {len(class_df)} samples")
                print(f"  Range: [{class_df[self.aggregated_var_name].min():.2f}, "
                      f"{class_df[self.aggregated_var_name].max():.2f}]")
        else:
            # Regression only - create dummy class column (not used, but needed for consistency)
            self.df[self.class_var_name] = 0
            self.target_classes = {}
            print("\n(Skipping classification class creation - regression only mode)")

        print("="*80 + "\n")

        return self.df, self.target_classes

    def _get_aggregation_formula(self):
        """
        Get human-readable formula for current aggregation.

        Returns:
            str: Mathematical formula using actual column names
        """
        # Handle single column case (no aggregation)
        if self.aggregation_cols and len(self.aggregation_cols) == 1:
            col = self.aggregation_cols[0]
            return f"target = {col} (no aggregation)"

        # Use actual column names or generic names for multi-column
        if self.aggregation_cols and len(self.aggregation_cols) >= 2:
            a = self.aggregation_cols[0]
            b = self.aggregation_cols[1]
        else:
            a, b = 'col1', 'col2'

        formulas = {
            'sum': f'{self.aggregation_name}({a}, {b}) = {a} + {b}',
            'mean': f'{self.aggregation_name}({a}, {b}) = ({a} + {b}) / 2',
            'max': f'{self.aggregation_name}({a}, {b}) = max({a}, {b})',
            'euclidean': f'{self.aggregation_name}({a}, {b}) = sqrt({a}² + {b}²)',
            'manhattan': f'{self.aggregation_name}({a}, {b}) = |{a}| + |{b}|',
            'absolute_diff': f'{self.aggregation_name}({a}, {b}) = |{a} - {b}|',
            'harmonic_mean': f'{self.aggregation_name}({a}, {b}) = 2 / (1/{a} + 1/{b})',
            'geometric_mean': f'{self.aggregation_name}({a}, {b}) = sqrt({a} * {b})',
            'weighted': f'{self.aggregation_name}({a}, {b}) = 0.6*{a} + 0.4*{b}',
            'weighted_70_30': f'{self.aggregation_name}({a}, {b}) = 0.7*{a} + 0.3*{b}',
            'rms': f'{self.aggregation_name}({a}, {b}) = sqrt(({a}² + {b}²) / 2)',
            'power_mean_3': f'{self.aggregation_name}({a}, {b}) = cbrt({a}³ + {b}³)',
            'chebyshev': f'{self.aggregation_name}({a}, {b}) = max(|{a}|, |{b}|)',
            'minkowski': f'{self.aggregation_name}({a}, {b}) = (|{a}|³ + |{b}|³)^(1/3)',
            'seuclidean': f'{self.aggregation_name}({a}, {b}) = sqrt({a}²/var({a}) + {b}²/var({b}))',
            'mahalanobis': f'{self.aggregation_name}({a}, {b}) = sqrt([{a},{b}]ᵀ * Σ⁻¹ * [{a},{b}])',
            'custom': f'custom({a}, {b}) = user-defined function',
        }
        return formulas.get(self.aggregation_name, f'{self.aggregation_name}({a}, {b})')

    def save_preprocessed_data(self, output_path):
        """
        Save preprocessed DataFrame to CSV

        Args:
            output_path: Path where to save the file

        Returns:
            Path: Full path to saved file

        Example:
            >>> preprocessor.save_preprocessed_data('./output/data/preprocessed/data.csv')
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        print(f"✓ Preprocessed data saved to: {output_path}")

        return output_path

    def _handle_non_numeric_columns(self, df, encoding_threshold=10, encoding_method='label', exclude_cols=None):
        """
        Handle non-numeric columns by converting dates to features and encoding categoricals.

        - Excluded columns (from EXCLUDE_COLS): Dropped without processing
        - Date/datetime columns: Extract year, month, day, dayofweek, is_weekend
        - Categorical columns (unique values <= threshold): Encode using label or onehot encoding
        - High-cardinality columns (unique values > threshold): Drop

        Args:
            df: DataFrame with feature columns
            encoding_threshold: Max unique values for a column to be encoded (default: 10)
            encoding_method: 'label' for LabelEncoder or 'onehot' for OneHotEncoding (default: 'label')
            exclude_cols: List of column names to exclude from processing (dropped directly)

        Returns:
            DataFrame with only numeric columns (dates and categoricals converted)
        """
        df_processed = df.copy()
        cols_to_drop = []
        encoded_cols = []

        # Get columns to exclude (from parameter or empty list)
        exclude_set = set(exclude_cols) if exclude_cols else set()

        for col in df_processed.columns:
            # Skip excluded columns - drop them directly without any processing
            if col in exclude_set:
                print(f"  Dropping excluded column '{col}' (in EXCLUDE_COLS)")
                cols_to_drop.append(col)
                continue

            dtype = df_processed[col].dtype

            # Check if column is object type (string/date/categorical)
            if dtype == 'object':
                # Try to parse as datetime first (suppress dateutil warning)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='Could not infer format')
                        date_col = pd.to_datetime(df_processed[col], errors='coerce')
                    # If more than 50% are valid dates, treat as date column
                    if date_col.notna().mean() > 0.5:
                        print(f"  Converting date column '{col}' to numeric features...")
                        df_processed[f'{col}_year'] = date_col.dt.year
                        df_processed[f'{col}_month'] = date_col.dt.month
                        df_processed[f'{col}_day'] = date_col.dt.day
                        df_processed[f'{col}_dayofweek'] = date_col.dt.dayofweek
                        df_processed[f'{col}_is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
                        cols_to_drop.append(col)
                        continue
                except Exception:
                    pass

                # Not a date - check if it's a categorical with limited unique values
                n_unique = df_processed[col].nunique()

                if n_unique <= encoding_threshold:
                    # Encode categorical column
                    if encoding_method == 'label':
                        print(f"  Encoding categorical column '{col}' ({n_unique} unique values) using LabelEncoder...")
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        encoded_cols.append(col)
                    elif encoding_method == 'onehot':
                        print(f"  Encoding categorical column '{col}' ({n_unique} unique values) using OneHotEncoding...")
                        # Create dummy variables
                        dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                        df_processed = pd.concat([df_processed, dummies], axis=1)
                        cols_to_drop.append(col)
                        encoded_cols.extend(dummies.columns.tolist())
                else:
                    # Too many unique values - drop the column
                    print(f"  Dropping high-cardinality column '{col}' ({n_unique} unique values > threshold {encoding_threshold})")
                    cols_to_drop.append(col)

            # Handle datetime columns directly
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                print(f"  Converting datetime column '{col}' to numeric features...")
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
                df_processed[f'{col}_is_weekend'] = (df_processed[col].dt.dayofweek >= 5).astype(int)
                cols_to_drop.append(col)

        # Drop original date/high-cardinality columns
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)

        if encoded_cols:
            print(f"  Encoded columns: {encoded_cols}")

        return df_processed

    def get_feature_target_split(self):
        """
        Split data into features and targets for both classification and regression

        Args:
            target_column: Column name for classification target
            regression_target: Column name for regression target

        Returns:
            dict: Dictionary containing X, y_classification, y_regression

        Example:
            >>> data = preprocessor.get_feature_target_split()
            >>> X = data['X']
            >>> y_clf = data['y_classification']
            >>> y_reg = data['y_regression']
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Define feature columns (exclude target variables, aggregation columns, and user-specified columns)
        exclude_cols = [self.class_var_name, self.aggregated_var_name] + list(self.aggregation_cols or [])

        # Add user-specified columns to exclude from Config
        user_exclude_cols = list(Config.EXCLUDE_COLS) if Config.EXCLUDE_COLS else []
        exclude_cols.extend(user_exclude_cols)

        feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        X = self.df[feature_cols]
        y_classification = self.df[self.class_var_name]
        y_regression = self.df[self.aggregated_var_name]

        print("="*80)
        print(" "*20 + "FEATURE-TARGET SPLIT")
        print("="*80)
        print(f"\nOriginal Features (X): {X.shape}")
        print(f"  Columns: {feature_cols}")
        if self.aggregation_cols:
            print(f"  (Excluded aggregation columns: {list(self.aggregation_cols)})")
        if user_exclude_cols:
            print(f"  (Excluded user-specified columns: {user_exclude_cols})")

        # Handle non-numeric columns (dates, categoricals, etc.)
        # Use config settings for encoding threshold and method
        # Pass exclude_cols as safety check (columns should already be excluded, but this ensures they're dropped)
        X = self._handle_non_numeric_columns(
            X,
            encoding_threshold=Config.CATEGORICAL_ENCODING_THRESHOLD,
            encoding_method=Config.CATEGORICAL_ENCODING_METHOD,
            exclude_cols=user_exclude_cols
        )
        final_feature_cols = X.columns.tolist()

        # Reset indices to ensure alignment between X and y
        X = X.reset_index(drop=True)
        y_classification = y_classification.reset_index(drop=True)
        y_regression = y_regression.reset_index(drop=True)

        print(f"\nProcessed Features (X): {X.shape}")
        print(f"  Columns: {final_feature_cols}")
        print(f"\nClassification Target (y): {y_classification.shape}")
        print(f"  Classes: {sorted(y_classification.unique())}")
        print(f"\nRegression Target (y): {y_regression.shape}")
        print(f"  Range: [{y_regression.min():.2f}, {y_regression.max():.2f}]")
        print("="*80 + "\n")

        return {
            'X': X,
            'y_classification': y_classification,
            'y_regression': y_regression,
            'feature_names': final_feature_cols
        }

    def get_multioutput_targets(self):
        """
        Get multi-output regression targets of target_vars

        Returns:
            numpy.ndarray: Array of shape (n_samples, n) with n columns of target variables

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> preprocessor.load_and_clean()
            >>> y_multi = preprocessor.get_multioutput_targets()
            >>> print(y_multi.shape)  # (n_samples, 2)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        n_cols = len(self.aggregation_cols) if self.aggregation_cols else 0

        if n_cols < 2:
            raise ValueError("At least two aggregation columns are required for multi-output targets. "
                           f"Current columns: {self.aggregation_cols}")

        # Use all aggregation columns for multi-output
        y_multi = self.df[self.aggregation_cols].values

        print("="*80)
        print(" "*20 + "MULTI-OUTPUT TARGETS")
        print("="*80)
        print(f"\nShape: {y_multi.shape}")
        for i, col in enumerate(self.aggregation_cols):
            print(f"  {col} range: [{y_multi[:, i].min():.2f}, {y_multi[:, i].max():.2f}]")
        print("="*80 + "\n")

        return y_multi
