"""
Tests for feature engineering utilities.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.features.utils import (
    normalize_features, create_interaction_features, create_polynomial_features,
    calculate_rolling_averages, calculate_trend_features, create_lag_features,
    calculate_percentile_features, detect_outliers, create_binned_features
)


class TestFeatureUtils:
    """Tests for feature engineering utility functions."""
    
    def test_normalize_features_with_default_ranges(self):
        """Test feature normalization with default ranges."""
        features = {
            'avg_finish_position': 10.0,  # Range 1-20, should be (10-1)/(20-1) = 0.474
            'podium_rate': 0.3,  # Range 0-1, should be 0.3
            'points_per_race': 12.5,  # Range 0-25, should be 0.5
            'unknown_feature': 42.0  # No range, should remain 42.0
        }
        
        normalized = normalize_features(features)
        
        assert abs(normalized['avg_finish_position'] - 0.474) < 0.01
        assert normalized['podium_rate'] == 0.3
        assert normalized['points_per_race'] == 0.5
        assert normalized['unknown_feature'] == 42.0
    
    def test_normalize_features_with_custom_ranges(self):
        """Test feature normalization with custom ranges."""
        features = {
            'custom_feature': 75.0,
            'another_feature': 150.0
        }
        
        custom_ranges = {
            'custom_feature': (50, 100),  # Should be (75-50)/(100-50) = 0.5
            'another_feature': (100, 200)  # Should be (150-100)/(200-100) = 0.5
        }
        
        normalized = normalize_features(features, custom_ranges)
        
        assert normalized['custom_feature'] == 0.5
        assert normalized['another_feature'] == 0.5
    
    def test_normalize_features_clipping(self):
        """Test that normalization clips values to [0, 1] range."""
        features = {
            'avg_finish_position': -5.0,  # Below range, should clip to 0
            'podium_rate': 1.5  # Above range, should clip to 1
        }
        
        normalized = normalize_features(features)
        
        assert normalized['avg_finish_position'] == 0.0
        assert normalized['podium_rate'] == 1.0
    
    def test_create_interaction_features(self):
        """Test creation of interaction features."""
        features = {
            'recent_form': 0.8,
            'constructor_performance': 0.7,
            'qualifying_delta': 0.2,
            'track_experience': 10.0,
            'temperature': 25.0,
            'rainfall': 0.0,
            'grid_position': 3.0
        }
        
        enhanced = create_interaction_features(features)
        
        # Check that original features are preserved
        for key, value in features.items():
            assert enhanced[key] == value
        
        # Check interaction features
        assert 'form_constructor_interaction' in enhanced
        assert enhanced['form_constructor_interaction'] == 0.8 * 0.7
        
        assert 'qualifying_experience_interaction' in enhanced
        assert enhanced['qualifying_experience_interaction'] == 0.2 * 10.0
        
        assert 'weather_interaction' in enhanced
        assert enhanced['weather_interaction'] == 25.0 * 0.0
    
    def test_create_interaction_features_missing_features(self):
        """Test interaction features when some base features are missing."""
        features = {
            'recent_form': 0.8,
            # Missing constructor_performance
        }
        
        enhanced = create_interaction_features(features)
        
        # Should not create interaction if one feature is missing
        assert 'form_constructor_interaction' not in enhanced
        assert enhanced['recent_form'] == 0.8
    
    def test_create_polynomial_features(self):
        """Test creation of polynomial features."""
        features = {
            'recent_form': 0.8,
            'constructor_performance': 0.6,
            'non_polynomial_feature': 1.0
        }
        
        enhanced = create_polynomial_features(features, degree=3)
        
        # Check original features preserved
        assert enhanced['recent_form'] == 0.8
        
        # Check polynomial features
        assert 'recent_form_poly_2' in enhanced
        assert enhanced['recent_form_poly_2'] == 0.8 ** 2
        
        assert 'recent_form_poly_3' in enhanced
        assert enhanced['recent_form_poly_3'] == 0.8 ** 3
        
        assert 'constructor_performance_poly_2' in enhanced
        assert enhanced['constructor_performance_poly_2'] == 0.6 ** 2
        
        # Non-polynomial candidates should not have polynomial features
        assert 'non_polynomial_feature_poly_2' not in enhanced
    
    def test_calculate_rolling_averages(self):
        """Test rolling average calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        windows = [2, 3, 5]
        
        rolling_avgs = calculate_rolling_averages(values, windows)
        
        assert 'rolling_avg_2' in rolling_avgs
        assert rolling_avgs['rolling_avg_2'] == 4.5  # Average of [4.0, 5.0]
        
        assert 'rolling_avg_3' in rolling_avgs
        assert rolling_avgs['rolling_avg_3'] == 4.0  # Average of [3.0, 4.0, 5.0]
        
        assert 'rolling_avg_5' in rolling_avgs
        assert rolling_avgs['rolling_avg_5'] == 3.0  # Average of all values
    
    def test_calculate_rolling_averages_insufficient_data(self):
        """Test rolling averages with insufficient data."""
        values = [1.0, 2.0]
        windows = [3, 5]
        
        rolling_avgs = calculate_rolling_averages(values, windows)
        
        # Should use all available data when window is larger
        assert rolling_avgs['rolling_avg_3'] == 1.5
        assert rolling_avgs['rolling_avg_5'] == 1.5
    
    def test_calculate_rolling_averages_empty(self):
        """Test rolling averages with empty data."""
        values = []
        windows = [2, 3]
        
        rolling_avgs = calculate_rolling_averages(values, windows)
        
        assert rolling_avgs == {}
    
    def test_calculate_trend_features(self):
        """Test trend feature calculation."""
        # Increasing trend
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        trends = calculate_trend_features(values)
        
        assert 'linear_trend' in trends
        assert trends['linear_trend'] > 0  # Positive trend
        
        assert 'volatility' in trends
        assert trends['volatility'] > 0
        
        assert 'momentum' in trends
        assert trends['momentum'] > 0  # Recent values higher than early
        
        assert 'consistency' in trends
        assert 0 < trends['consistency'] <= 1
    
    def test_calculate_trend_features_insufficient_data(self):
        """Test trend features with insufficient data."""
        values = [1.0]
        
        trends = calculate_trend_features(values)
        
        assert trends == {}  # Should return empty for single value
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Most recent is 5.0
        lags = [1, 2, 3]
        
        lag_features = create_lag_features(values, lags)
        
        assert 'lag_1' in lag_features
        assert lag_features['lag_1'] == 4.0  # Previous value
        
        assert 'lag_2' in lag_features
        assert lag_features['lag_2'] == 3.0  # Two values back
        
        assert 'lag_3' in lag_features
        assert lag_features['lag_3'] == 2.0  # Three values back
    
    def test_create_lag_features_insufficient_data(self):
        """Test lag features with insufficient data."""
        values = [1.0, 2.0]
        lags = [1, 2, 5]  # lag=5 exceeds available data
        
        lag_features = create_lag_features(values, lags)
        
        assert 'lag_1' in lag_features
        assert lag_features['lag_1'] == 1.0
        
        assert 'lag_2' not in lag_features  # Not enough data
        assert 'lag_5' not in lag_features  # Not enough data
    
    def test_calculate_percentile_features(self):
        """Test percentile feature calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        percentiles = calculate_percentile_features(values)
        
        assert 'percentile_25' in percentiles
        assert percentiles['percentile_25'] == 3.25
        
        assert 'percentile_50' in percentiles
        assert percentiles['percentile_50'] == 5.5
        
        assert 'percentile_75' in percentiles
        assert percentiles['percentile_75'] == 7.75
    
    def test_calculate_percentile_features_custom_percentiles(self):
        """Test percentile features with custom percentiles."""
        values = [1, 2, 3, 4, 5]
        custom_percentiles = [10, 90]
        
        percentiles = calculate_percentile_features(values, custom_percentiles)
        
        assert 'percentile_10' in percentiles
        assert 'percentile_90' in percentiles
        assert 'percentile_25' not in percentiles  # Default not included
    
    def test_calculate_percentile_features_empty(self):
        """Test percentile features with empty data."""
        values = []
        
        percentiles = calculate_percentile_features(values)
        
        assert percentiles == {}
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Normal values with outliers
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        
        outlier_info = detect_outliers(values, method='iqr')
        
        assert outlier_info['outlier_count'] == 1
        assert 9 in outlier_info['outlier_indices']  # Index of value 100
        assert outlier_info['outlier_rate'] == 0.1
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using z-score method."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is an outlier
        
        outlier_info = detect_outliers(values, method='zscore')
        
        assert outlier_info['outlier_count'] >= 1
        assert outlier_info['outlier_rate'] > 0
    
    def test_detect_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        values = [1, 2]  # Too few values
        
        outlier_info = detect_outliers(values)
        
        assert outlier_info['outlier_count'] == 0
        assert outlier_info['outlier_indices'] == []
        assert outlier_info['outlier_rate'] == 0.0
    
    def test_create_binned_features(self):
        """Test creation of binned features."""
        features = {
            'recent_form': 0.7,  # Should fall in bin 1 (0.3-0.6 range)
            'track_experience': 12.0,  # Should fall in bin 1 (5-15 range)
            'unbinned_feature': 42.0
        }
        
        enhanced = create_binned_features(features)
        
        # Check original features preserved
        assert enhanced['recent_form'] == 0.7
        
        # Check binned features for recent_form (bins: [0, 0.3, 0.6, 1.0])
        # 0.7 falls in the last bin (0.6-1.0)
        assert 'recent_form_bin_0' in enhanced
        assert enhanced['recent_form_bin_0'] == 0.0
        
        assert 'recent_form_bin_1' in enhanced
        assert enhanced['recent_form_bin_1'] == 0.0
        
        assert 'recent_form_bin_2' in enhanced
        assert enhanced['recent_form_bin_2'] == 1.0
        
        # Check binned features for track_experience (bins: [0, 5, 15, 50])
        # 12.0 falls in bin 1 (5-15)
        assert 'track_experience_bin_1' in enhanced
        assert enhanced['track_experience_bin_1'] == 1.0
        
        # Unbinned features should not have bin features
        assert 'unbinned_feature_bin_0' not in enhanced
    
    def test_create_binned_features_custom_config(self):
        """Test binned features with custom configuration."""
        features = {
            'custom_feature': 7.5
        }
        
        custom_binning = {
            'custom_feature': [0, 5, 10, 15]  # 7.5 falls in bin 1 (5-10)
        }
        
        enhanced = create_binned_features(features, custom_binning)
        
        assert 'custom_feature_bin_0' in enhanced
        assert enhanced['custom_feature_bin_0'] == 0.0
        
        assert 'custom_feature_bin_1' in enhanced
        assert enhanced['custom_feature_bin_1'] == 1.0
        
        assert 'custom_feature_bin_2' in enhanced
        assert enhanced['custom_feature_bin_2'] == 0.0