"""
Data Preprocessing Module

This module handles data loading, preprocessing, and feature engineering.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.aggregations import get_aggregation_function
# Metrics for clustering evaluation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


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

            self.df = pd.read_csv(self.data_path)

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


    def create_target_variables(self, n_classes=4, clustering_method='qcut'):
        """
        Create aggregated target variable and classification classes

        Args:
            n_classes: Number of classes for classification (default: 4)
            clustering_method: Method to create classes - 'qcut' or 'kmeans' (default: 'qcut')
                'qcut': Quantile-based binning on aggregated values
                'kmeans': KMeans clustering on target_vars space

        Returns:
            tuple: (df, target_classes_dict) - Updated dataframe and dict of class DataFrames

        Example:
            >>> df, classes = preprocessor.create_target_variables(n_classes=4, clustering_method='kmeans')
        """
        print("="*80)
        print(" "*20 + "CREATING TARGET VARIABLES")
        print("="*80)

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get aggregation function
        aggregation_func = get_aggregation_function(self.aggregation_name)

        print(f"\nUsing aggregation: {self.aggregation_name}")
        print(f"Formula: {self._get_aggregation_formula()}")

        # Create aggregated target
        self.df[self.aggregated_var_name] = aggregation_func(
            self.df[self.aggregation_cols[0]],
            self.df[self.aggregation_cols[1]]
        )

        print("\nFirst 5 rows with charges_sum:")
        print(self.df[[self.aggregation_cols[0], self.aggregation_cols[1], self.aggregated_var_name]].head())

        # Create classification classes based on specified method
        print(f"\nClustering method: {clustering_method.upper()}")

        if clustering_method == 'kmeans':
            # Use KMeans clustering on target_vars space
            print(f"Using KMeans clustering to find {n_classes} natural clusters...")

            # Prepare data for clustering
            X = self.df[[self.aggregation_cols[0], self.aggregation_cols[1]]].values
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
            for i, (h, c) in enumerate(centers):
                print(f"    Cluster {i}: ({h:.2f}, {c:.2f})")

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

        print("="*80 + "\n")

        return self.df, self.target_classes

    def _get_aggregation_formula(self):
        """Get human-readable formula for current aggregation"""
        formulas = {
            'sum': '+ cooling_load',
            'mean': '+ cooling_load) / 2',
            'max': 'vs cooling_load (maximum)',
            'euclidean': '² + cooling_load² (square root)',
            'manhattan': '| + |cooling_load|',
            'absolute_diff': '- cooling_load| (absolute)',
            'harmonic_mean': 'and cooling_load (harmonic mean)',
            'geometric_mean': '* cooling_load (square root)',
            'weighted': '* 0.6 + cooling_load * 0.4',
            'weighted_70_30': '* 0.7 + cooling_load * 0.3',
            'rms': '² + cooling_load²) / 2 (square root)',
            'power_mean_3': '³ + cooling_load³ (cube root)',
        }
        return formulas.get(self.aggregation_name, 'custom aggregation')

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

        # Define feature columns (exclude target variables)
        exclude_cols = [self.class_var_name, self.aggregated_var_name]
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        X = self.df[feature_cols]
        y_classification = self.df[self.class_var_name]
        y_regression = self.df[self.aggregated_var_name]

        print("="*80)
        print(" "*20 + "FEATURE-TARGET SPLIT")
        print("="*80)
        print(f"\nFeatures (X): {X.shape}")
        print(f"  Columns: {feature_cols}")
        print(f"\nClassification Target (y): {y_classification.shape}")
        print(f"  Classes: {sorted(y_classification.unique())}")
        print(f"\nRegression Target (y): {y_regression.shape}")
        print(f"  Range: [{y_regression.min():.2f}, {y_regression.max():.2f}]")
        print("="*80 + "\n")

        return {
            'X': X,
            'y_classification': y_classification,
            'y_regression': y_regression,
            'feature_names': feature_cols
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

        if len(self.aggregation_cols) < 2:
            raise ValueError("At least two aggregation columns are required for multi-output targets")

        y_multi = self.df[[self.aggregation_cols[0], self.aggregation_cols[1]]].values

        print("="*80)
        print(" "*20 + "MULTI-OUTPUT TARGETS")
        print("="*80)
        print(f"\nShape: {y_multi.shape}")
        print(f"  {self.aggregation_cols[0]} range: [{y_multi[:, 0].min():.2f}, {y_multi[:, 0].max():.2f}]")
        print(f"  {self.aggregation_cols[1]} range: [{y_multi[:, 1].min():.2f}, {y_multi[:, 1].max():.2f}]")
        print("="*80 + "\n")

        return y_multi
