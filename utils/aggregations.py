"""
Custom Aggregation Functions for Target Variable Creation

This module provides various aggregation functions to combine heating_load and cooling_load
into a single target variable for regression and classification tasks.

Users can define custom aggregation functions in config.py by setting:
    CUSTOM_AGGREGATION_FUNCTION = lambda h, c: formula

Example:
    # Linear relationship: cooling = (1+a)*heating + b
    CUSTOM_AGGREGATION_FUNCTION = lambda h, c: (1 + 0.920) * h + 4.064
"""
import numpy as np


# Global variable to store user-defined custom aggregation
_custom_aggregation_function = None


def set_custom_aggregation(custom_function):
    """
    Set a user-defined custom aggregation function

    Args:
        custom_function: A callable that takes two arguments (heating, cooling)
                        and returns aggregated values

    Example:
        >>> # Linear relationship from correlation analysis
        >>> set_custom_aggregation(lambda h, c: (1 + 0.920) * h + 4.064)
    """
    global _custom_aggregation_function
    _custom_aggregation_function = custom_function
    print(f"Custom aggregation function registered: {custom_function}")


def _seuclidean_aggregation(h, c):
    """
    Standardized Euclidean distance aggregation.

    Formula: sqrt(((h-0)²/V[h]) + ((c-0)²/V[c]))
    where V is the variance vector.

    For aggregation, we compute the distance from origin (0,0) with
    standardization by the variance of each variable.

    Args:
        h: heating_load values (array-like)
        c: cooling_load values (array-like)

    Returns:
        Standardized Euclidean distance from origin
    """
    # Convert to numpy arrays
    h = np.asarray(h)
    c = np.asarray(c)

    # Compute variances (using the entire dataset)
    var_h = np.var(h) if np.var(h) > 0 else 1.0  # Avoid division by zero
    var_c = np.var(c) if np.var(c) > 0 else 1.0

    # Standardized Euclidean distance from origin
    return np.sqrt((h**2 / var_h) + (c**2 / var_c))


def _mahalanobis_aggregation(h, c):
    """
    Mahalanobis distance aggregation.

    Formula: sqrt((x - y)^T * V^-1 * (x - y))
    where V^-1 is the inverse covariance matrix.

    For aggregation, we compute the Mahalanobis distance from origin (0,0)
    using the covariance structure of the data.

    Args:
        h: heating_load values (array-like)
        c: cooling_load values (array-like)

    Returns:
        Mahalanobis distance from origin
    """
    # Convert to numpy arrays
    h = np.asarray(h)
    c = np.asarray(c)

    # Stack into matrix for covariance calculation
    data = np.column_stack([h, c])

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
    # For each point [h_i, c_i], compute sqrt([h_i, c_i]^T * inv_cov * [h_i, c_i])
    result = np.zeros(len(h))
    for i in range(len(h)):
        point = np.array([h[i], c[i]])
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
        callable: A function that takes two pandas Series/arrays (heating, cooling)
                 and returns aggregated values

    Raises:
        ValueError: If aggregation_type is not recognized or 'custom' is used
                   without setting a custom function

    Examples:
        >>> agg_func = get_aggregation_function('manhattan')
        >>> result = agg_func(heating_data, cooling_data)

        >>> # Using custom function
        >>> set_custom_aggregation(lambda h, c: (1 + 0.920) * h + 4.064)
        >>> agg_func = get_aggregation_function('custom')
        >>> result = agg_func(heating_data, cooling_data)
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
        'sum': lambda h, c: h + c,

        'mean': lambda h, c: 0.5 * (h + c),

        'max': lambda h, c: np.maximum(h, c),

        'euclidean': lambda h, c: np.sqrt(h**2 + c**2),

        'manhattan': lambda h, c: np.abs(h) + np.abs(c),

        'absolute_diff': lambda h, c: np.abs(h - c),

        'harmonic_mean': lambda h, c: 2 / (1/h + 1/c),

        'geometric_mean': lambda h, c: np.sqrt(h * c),

        'weighted': lambda h, c: 0.6 * h + 0.4 * c,

        'weighted_70_30': lambda h, c: 0.7 * h + 0.3 * c,

        'rms': lambda h, c: np.sqrt((h**2 + c**2) / 2),

        'power_mean_3': lambda h, c: np.cbrt(h**3 + c**3),

        # Distance metrics from scipy.spatial.distance
        'chebyshev': lambda h, c: np.maximum(np.abs(h), np.abs(c)),

        'minkowski': lambda h, c: np.power(np.abs(h)**3 + np.abs(c)**3, 1/3),

        'seuclidean': lambda h, c: _seuclidean_aggregation(h, c),

        'mahalanobis': lambda h, c: _mahalanobis_aggregation(h, c),
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
        'sum': 'Simple sum: h + c',
        'mean': 'Arithmetic mean: 0.5 * (h + c)',
        'max': 'Maximum value: max(h, c)',
        'euclidean': 'Euclidean distance: sqrt(h² + c²)',
        'manhattan': 'Manhattan distance: |h| + |c|',
        'absolute_diff': 'Absolute difference: |h - c|',
        'harmonic_mean': 'Harmonic mean: 2 / (1/h + 1/c)',
        'geometric_mean': 'Geometric mean: sqrt(h * c)',
        'weighted': 'Weighted average (60-40): 0.6*h + 0.4*c',
        'weighted_70_30': 'Weighted average (70-30): 0.7*h + 0.3*c',
        'rms': 'Root Mean Square: sqrt((h² + c²) / 2)',
        'power_mean_3': 'Power mean (p=3): (h³ + c³)^(1/3)',
        'chebyshev': 'Chebyshev distance: max(|h|, |c|)',
        'minkowski': 'Minkowski distance (p=3): (|h|³ + |c|³)^(1/3)',
        'seuclidean': 'Standardized Euclidean: sqrt(h²/var(h) + c²/var(c))',
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
