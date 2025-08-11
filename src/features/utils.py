"""
Utility functions for feature engineering.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..models.data_models import RaceData, RaceResult


def normalize_features(features: Dict[str, float], 
                      feature_ranges: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
    """
    Normalize features to [0, 1] range.
    
    Args:
        features: Dictionary of feature values
        feature_ranges: Optional dictionary of (min, max) ranges for each feature
        
    Returns:
        Dictionary of normalized features
    """
    normalized = {}
    
    # Default ranges for common F1 features
    default_ranges = {
        'avg_finish_position': (1, 20),
        'avg_grid_position': (1, 20),
        'points_per_race': (0, 25),
        'podium_rate': (0, 1),
        'finish_rate': (0, 1),
        'dnf_rate': (0, 1),
        'recent_form': (0, 1),
        'constructor_performance': (0, 1),
        'qualifying_delta': (-1, 1),
        'track_experience': (0, 50),
        'championship_position': (1, 20),
        'temperature': (-10, 50),
        'humidity': (0, 100),
        'wind_speed': (0, 50)
    }
    
    ranges = feature_ranges or default_ranges
    
    for feature_name, value in features.items():
        if feature_name in ranges:
            min_val, max_val = ranges[feature_name]
            normalized_value = (value - min_val) / (max_val - min_val)
            normalized[feature_name] = max(0, min(1, normalized_value))
        else:
            # If no range specified, assume already normalized or leave as-is
            normalized[feature_name] = value
    
    return normalized


def calculate_feature_importance_scores(race_results: List[RaceData]) -> Dict[str, float]:
    """
    Calculate simple feature importance based on correlation with race results.
    
    Args:
        race_results: List of race data
        
    Returns:
        Dictionary of feature importance scores
    """
    if not race_results:
        return {}
    
    # Collect data for correlation analysis
    data_points = []
    
    for race in race_results:
        for result in race.results:
            if result.final_position < 999:  # Only finished races
                data_point = {
                    'final_position': result.final_position,
                    'grid_position': result.grid_position,
                    'points': result.points,
                    'constructor_id': result.constructor_id,
                    'temperature': race.weather.temperature,
                    'humidity': race.weather.humidity,
                    'rainfall': 1.0 if race.weather.rainfall else 0.0
                }
                data_points.append(data_point)
    
    if len(data_points) < 10:  # Need minimum data for meaningful correlation
        return {}
    
    # Convert to DataFrame for correlation analysis
    df = pd.DataFrame(data_points)
    
    # Calculate correlations with final position (lower is better)
    correlations = {}
    target = df['final_position']
    
    for column in df.columns:
        if column != 'final_position' and df[column].dtype in ['int64', 'float64']:
            corr = abs(df[column].corr(target))
            if not np.isnan(corr):
                correlations[column] = corr
    
    return correlations


def create_interaction_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Create interaction features between existing features.
    
    Args:
        features: Dictionary of base features
        
    Returns:
        Dictionary with original features plus interaction features
    """
    enhanced_features = features.copy()
    
    # Define meaningful interactions for F1 racing
    interactions = [
        ('recent_form', 'constructor_performance', 'form_constructor_interaction'),
        ('qualifying_delta', 'track_experience', 'qualifying_experience_interaction'),
        ('temperature', 'rainfall', 'weather_interaction'),
        ('grid_position', 'recent_form', 'grid_form_interaction'),
        ('constructor_performance', 'track_experience', 'constructor_experience_interaction')
    ]
    
    for feature1, feature2, interaction_name in interactions:
        if feature1 in features and feature2 in features:
            # Simple multiplication interaction
            enhanced_features[interaction_name] = features[feature1] * features[feature2]
    
    return enhanced_features


def create_polynomial_features(features: Dict[str, float], degree: int = 2) -> Dict[str, float]:
    """
    Create polynomial features for specified numeric features.
    
    Args:
        features: Dictionary of base features
        degree: Polynomial degree (2 for quadratic, 3 for cubic, etc.)
        
    Returns:
        Dictionary with original features plus polynomial features
    """
    enhanced_features = features.copy()
    
    # Features that might benefit from polynomial transformation
    polynomial_candidates = [
        'recent_form', 'constructor_performance', 'qualifying_delta',
        'track_experience', 'temperature', 'humidity'
    ]
    
    for feature_name in polynomial_candidates:
        if feature_name in features:
            value = features[feature_name]
            
            for d in range(2, degree + 1):
                poly_feature_name = f"{feature_name}_poly_{d}"
                enhanced_features[poly_feature_name] = value ** d
    
    return enhanced_features


def calculate_rolling_averages(values: List[float], windows: List[int]) -> Dict[str, float]:
    """
    Calculate rolling averages for different window sizes.
    
    Args:
        values: List of values (most recent last)
        windows: List of window sizes
        
    Returns:
        Dictionary of rolling averages
    """
    rolling_avgs = {}
    
    if not values:
        return rolling_avgs
    
    for window in windows:
        if window <= len(values):
            recent_values = values[-window:]
            avg = np.mean(recent_values)
            rolling_avgs[f'rolling_avg_{window}'] = avg
        else:
            # If not enough data, use all available
            rolling_avgs[f'rolling_avg_{window}'] = np.mean(values)
    
    return rolling_avgs


def calculate_trend_features(values: List[float], timestamps: List[datetime] = None) -> Dict[str, float]:
    """
    Calculate trend-based features from a time series of values.
    
    Args:
        values: List of values (chronologically ordered)
        timestamps: Optional list of timestamps
        
    Returns:
        Dictionary of trend features
    """
    trend_features = {}
    
    if len(values) < 2:
        return trend_features
    
    # Simple linear trend
    x = np.arange(len(values))
    y = np.array(values)
    
    # Calculate slope (trend direction)
    if len(values) > 1:
        slope = np.polyfit(x, y, 1)[0]
        trend_features['linear_trend'] = slope
    
    # Calculate volatility (standard deviation)
    if len(values) > 1:
        trend_features['volatility'] = np.std(values)
    
    # Calculate momentum (recent vs early performance)
    if len(values) >= 4:
        recent_avg = np.mean(values[-2:])
        early_avg = np.mean(values[:2])
        trend_features['momentum'] = recent_avg - early_avg
    
    # Calculate consistency (inverse of coefficient of variation)
    if len(values) > 1 and np.mean(values) != 0:
        cv = np.std(values) / abs(np.mean(values))
        trend_features['consistency'] = 1 / (1 + cv)  # Higher is more consistent
    
    return trend_features


def create_lag_features(values: List[float], lags: List[int]) -> Dict[str, float]:
    """
    Create lag features from a time series.
    
    Args:
        values: List of values (most recent last)
        lags: List of lag periods
        
    Returns:
        Dictionary of lag features
    """
    lag_features = {}
    
    for lag in lags:
        if lag < len(values):
            lag_value = values[-(lag + 1)]  # lag=1 means previous value
            lag_features[f'lag_{lag}'] = lag_value
    
    return lag_features


def calculate_percentile_features(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
    """
    Calculate percentile-based features.
    
    Args:
        values: List of values
        percentiles: List of percentiles to calculate (default: [25, 50, 75])
        
    Returns:
        Dictionary of percentile features
    """
    if percentiles is None:
        percentiles = [25, 50, 75]
    
    percentile_features = {}
    
    if not values:
        return percentile_features
    
    for p in percentiles:
        percentile_value = np.percentile(values, p)
        percentile_features[f'percentile_{p}'] = percentile_value
    
    return percentile_features


def detect_outliers(values: List[float], method: str = 'iqr') -> Dict[str, Any]:
    """
    Detect outliers in a list of values.
    
    Args:
        values: List of values
        method: Outlier detection method ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier information
    """
    outlier_info = {
        'outlier_count': 0,
        'outlier_indices': [],
        'outlier_rate': 0.0
    }
    
    if len(values) < 4:  # Need minimum data for outlier detection
        return outlier_info
    
    values_array = np.array(values)
    
    if method == 'iqr':
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (values_array < lower_bound) | (values_array > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((values_array - np.mean(values_array)) / np.std(values_array))
        outliers = z_scores > 2.5  # 2.5 standard deviations
    
    else:
        return outlier_info
    
    outlier_indices = np.where(outliers)[0].tolist()
    
    outlier_info.update({
        'outlier_count': len(outlier_indices),
        'outlier_indices': outlier_indices,
        'outlier_rate': len(outlier_indices) / len(values)
    })
    
    return outlier_info


def create_binned_features(features: Dict[str, float], 
                          binning_config: Dict[str, List[float]] = None) -> Dict[str, float]:
    """
    Create binned (discretized) versions of continuous features.
    
    Args:
        features: Dictionary of features
        binning_config: Dictionary mapping feature names to bin edges
        
    Returns:
        Dictionary with original features plus binned features
    """
    enhanced_features = features.copy()
    
    # Default binning configuration for F1 features
    default_binning = {
        'recent_form': [0, 0.3, 0.6, 1.0],  # Low, Medium, High
        'track_experience': [0, 5, 15, 50],  # Rookie, Experienced, Veteran
        'championship_position': [1, 3, 10, 20],  # Top, Midfield, Back
        'temperature': [0, 15, 25, 50]  # Cold, Mild, Warm, Hot
    }
    
    binning = binning_config or default_binning
    
    for feature_name, bin_edges in binning.items():
        if feature_name in features:
            value = features[feature_name]
            
            # Find which bin the value falls into
            bin_index = np.digitize(value, bin_edges) - 1
            bin_index = max(0, min(bin_index, len(bin_edges) - 2))
            
            # Create one-hot encoded bins
            for i in range(len(bin_edges) - 1):
                bin_feature_name = f"{feature_name}_bin_{i}"
                enhanced_features[bin_feature_name] = 1.0 if i == bin_index else 0.0
    
    return enhanced_features