"""
Custom Aggregation Functions for Target Variable Creation

This module provides various aggregation functions to combine two columns
into a single target variable for regression and classification tasks.

Users can define custom aggregation functions in config.py by setting:
    CUSTOM_AGGREGATION_FUNCTION = lambda col1, col2: formula

Example:
    # Linear relationship
    CUSTOM_AGGREGATION_FUNCTION = lambda col1, col2: (1 + 0.920) * col1 + 4.064
"""
import numpy as np


# Global variable to store user-defined custom aggregation
_custom_aggregation_function = None


def set_custom_aggregation(custom_function):
    """
    Set a user-defined custom aggregation function

    Args:
        custom_function: A callable that takes two arguments (col1, col2)
                        and returns aggregated values

    Example:
        >>> # Linear relationship from correlation analysis
        >>> set_custom_aggregation(lambda col1, col2: (1 + 0.920) * col1 + 4.064)
    """
    global _custom_aggregation_function
    _custom_aggregation_function = custom_function
    print(f"Custom aggregation function registered: {custom_function}")


def _seuclidean_aggregation(col1, col2):
    """
    Standardized Euclidean distance aggregation.

    Formula: sqrt(((col1-0)²/V[col1]) + ((col2-0)²/V[col2]))
    where V is the variance vector.

    For aggregation, we compute the distance from origin (0,0) with
    standardization by the variance of each variable.

    Args:
        col1: First column values (array-like)
        col2: Second column values (array-like)

    Returns:
        Standardized Euclidean distance from origin
    """
    # Convert to numpy arrays
    col1 = np.asarray(col1)
    col2 = np.asarray(col2)

    # Compute variances (using the entire dataset)
    var_col1 = np.var(col1) if np.var(col1) > 0 else 1.0  # Avoid division by zero
    var_col2 = np.var(col2) if np.var(col2) > 0 else 1.0

    # Standardized Euclidean distance from origin
    return np.sqrt((col1**2 / var_col1) + (col2**2 / var_col2))


def _mahalanobis_aggregation(col1, col2):
    """
    Mahalanobis distance aggregation.

    Formula: sqrt((x - y)^T * V^-1 * (x - y))
    where V^-1 is the inverse covariance matrix.

    For aggregation, we compute the Mahalanobis distance from origin (0,0)
    using the covariance structure of the data.

    Args:
        col1: First column values (array-like)
        col2: Second column values (array-like)

    Returns:
        Mahalanobis distance from origin
    """
    # Convert to numpy arrays
    col1 = np.asarray(col1)
    col2 = np.asarray(col2)

    # Stack into matrix for covariance calculation
    data = np.column_stack([col1, col2])

    # Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False)

    # Handle singular matrix (add small regularization if needed)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add regularization term if matrix is singular
        cov_matrix += np.eye(2) * 1e-6
        inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distance for each point from origin
    # For each point [col1_i, col2_i], compute sqrt([col1_i, col2_i]^T * inv_cov * [col1_i, col2_i])
    result = np.zeros(len(col1))
    for i in range(len(col1)):
        point = np.array([col1[i], col2[i]])
        result[i] = np.sqrt(point.T @ inv_cov_matrix @ point)

    return result


def get_aggregation_function(aggregation_type='sum'):
    """
    Get aggregation function by name

    Args:
        aggregation_type: Type of aggregation function
                         Options: 'sum', 'mean', 'max', 'euclidean', 'manhattan',
                                 'absolute_diff', 'harmonic_mean', 'geometric_mean',
                                 'weighted', 'weighted_70_30', 'rms', 'power_mean_3',
                                 'chebyshev', 'minkowski', 'seuclidean', 'mahalanobis',
                                 'custom' (user-defined function from config.py)

    Returns:
        callable: A function that takes two pandas Series/arrays (col1, col2)
                 and returns aggregated values

    Raises:
        ValueError: If aggregation_type is not recognized or 'custom' is used
                   without setting a custom function

    Examples:
        >>> agg_func = get_aggregation_function('manhattan')
        >>> result = agg_func(column1_data, column2_data)

        >>> # Using custom function
        >>> set_custom_aggregation(lambda col1, col2: (1 + 0.920) * col1 + 4.064)
        >>> agg_func = get_aggregation_function('custom')
        >>> result = agg_func(column1_data, column2_data)
    """

    # Check if custom aggregation is requested
    if aggregation_type == 'custom':
        if _custom_aggregation_function is None:
            raise ValueError(
                "Custom aggregation requested but no custom function defined. "
                "Please set CUSTOM_AGGREGATION_FUNCTION in config.py or call "
                "set_custom_aggregation() first."
            )
        return _custom_aggregation_function

    aggregations = {
        'sum': lambda col1, col2: col1 + col2,

        'mean': lambda col1, col2: 0.5 * (col1 + col2),

        'max': lambda col1, col2: np.maximum(col1, col2),

        'euclidean': lambda col1, col2: np.sqrt(col1**2 + col2**2),

        'manhattan': lambda col1, col2: np.abs(col1) + np.abs(col2),

        'absolute_diff': lambda col1, col2: np.abs(col1 - col2),

        'harmonic_mean': lambda col1, col2: 2 / (1/col1 + 1/col2),

        'geometric_mean': lambda col1, col2: np.sqrt(col1 * col2),

        'weighted': lambda col1, col2: 0.6 * col1 + 0.4 * col2,

        'weighted_70_30': lambda col1, col2: 0.7 * col1 + 0.3 * col2,

        'rms': lambda col1, col2: np.sqrt((col1**2 + col2**2) / 2),

        'power_mean_3': lambda col1, col2: np.cbrt(col1**3 + col2**3),

        # Distance metrics from scipy.spatial.distance
        'chebyshev': lambda col1, col2: np.maximum(np.abs(col1), np.abs(col2)),

        'minkowski': lambda col1, col2: np.power(np.abs(col1)**3 + np.abs(col2)**3, 1/3),

        'seuclidean': lambda col1, col2: _seuclidean_aggregation(col1, col2),

        'mahalanobis': lambda col1, col2: _mahalanobis_aggregation(col1, col2),
    }

    if aggregation_type not in aggregations:
        available = ', '.join(f"'{k}'" for k in list(aggregations.keys()) + ['custom'])
        raise ValueError(
            f"Unknown aggregation_type: '{aggregation_type}'. "
            f"Choose from: {available}"
        )

    return aggregations[aggregation_type]


def list_available_aggregations():
    """
    List all available aggregation functions with descriptions

    Returns:
        dict: Dictionary mapping aggregation names to descriptions
    """
    aggregations = {
        'sum': 'Simple sum: col1 + col2',
        'mean': 'Arithmetic mean: 0.5 * (col1 + col2)',
        'max': 'Maximum value: max(col1, col2)',
        'euclidean': 'Euclidean distance: sqrt(col1² + col2²)',
        'manhattan': 'Manhattan distance: |col1| + |col2|',
        'absolute_diff': 'Absolute difference: |col1 - col2|',
        'harmonic_mean': 'Harmonic mean: 2 / (1/col1 + 1/col2)',
        'geometric_mean': 'Geometric mean: sqrt(col1 * col2)',
        'weighted': 'Weighted average (60-40): 0.6*col1 + 0.4*col2',
        'weighted_70_30': 'Weighted average (70-30): 0.7*col1 + 0.3*col2',
        'rms': 'Root Mean Square: sqrt((col1² + col2²) / 2)',
        'power_mean_3': 'Power mean (p=3): (col1³ + col2³)^(1/3)',
        'chebyshev': 'Chebyshev distance: max(|col1|, |col2|)',
        'minkowski': 'Minkowski distance (p=3): (|col1|³ + |col2|³)^(1/3)',
        'seuclidean': 'Standardized Euclidean: sqrt(col1²/var(col1) + col2²/var(col2))',
        'mahalanobis': 'Mahalanobis distance: sqrt(x^T * Σ^-1 * x)',
        'custom': 'User-defined function (set in config.py)',
    }
    return aggregations


def print_aggregation_info():
    """Print information about all available aggregations"""
    print("\n" + "="*70)
    print(" "*20 + "AVAILABLE AGGREGATIONS")
    print("="*70)

    aggregations = list_available_aggregations()
    for name, description in aggregations.items():
        print(f"  {name:20s} : {description}")

    print("="*70 + "\n")


# Backward compatibility function name
def target_variable_custom_aggregation(aggregation_type='euclidean'):
    """
    Legacy function name for backward compatibility

    This is an alias for get_aggregation_function()
    """
    return get_aggregation_function(aggregation_type)
