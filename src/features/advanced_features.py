"""
Advanced feature engineering for F1 race prediction.
Combines base features, rolling statistics, and domain-specific features.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict

from .base_extractor import BaseFeatureExtractor
from .rolling_stats import RollingStatsCalculator
from .utils import (
    normalize_features, create_interaction_features, create_polynomial_features,
    calculate_trend_features, create_binned_features, detect_outliers
)
from ..models.data_models import RaceData, RaceResult, QualifyingResult, WeatherData
from ..config import config


logger = logging.getLogger(__name__)


class AdvancedFeatureEngineeringError(Exception):
    """Custom exception for advanced feature engineering errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class TrackSpecificFeatures:
    """
    Track-specific feature engineering for circuit characteristics.
    """
    
    def __init__(self):
        # Track characteristics database
        self.track_characteristics = {
            'monaco': {
                'overtaking_difficulty': 0.9,
                'street_circuit': 1.0,
                'elevation_change': 0.3,
                'corner_count': 19,
                'straight_length': 0.2,
                'weather_sensitivity': 0.8,
                'tire_degradation': 0.4,
                'safety_car_probability': 0.7
            },
            'monza': {
                'overtaking_difficulty': 0.2,
                'street_circuit': 0.0,
                'elevation_change': 0.1,
                'corner_count': 11,
                'straight_length': 0.9,
                'weather_sensitivity': 0.3,
                'tire_degradation': 0.6,
                'safety_car_probability': 0.3
            },
            'silverstone': {
                'overtaking_difficulty': 0.4,
                'street_circuit': 0.0,
                'elevation_change': 0.2,
                'corner_count': 18,
                'straight_length': 0.6,
                'weather_sensitivity': 0.9,
                'tire_degradation': 0.7,
                'safety_car_probability': 0.2
            },
            'spa': {
                'overtaking_difficulty': 0.3,
                'street_circuit': 0.0,
                'elevation_change': 0.8,
                'corner_count': 19,
                'straight_length': 0.8,
                'weather_sensitivity': 0.9,
                'tire_degradation': 0.8,
                'safety_car_probability': 0.4
            },
            'suzuka': {
                'overtaking_difficulty': 0.6,
                'street_circuit': 0.0,
                'elevation_change': 0.5,
                'corner_count': 18,
                'straight_length': 0.4,
                'weather_sensitivity': 0.7,
                'tire_degradation': 0.6,
                'safety_car_probability': 0.3
            }
        }
        
        # Default characteristics for unknown tracks
        self.default_characteristics = {
            'overtaking_difficulty': 0.5,
            'street_circuit': 0.0,
            'elevation_change': 0.3,
            'corner_count': 16,
            'straight_length': 0.5,
            'weather_sensitivity': 0.5,
            'tire_degradation': 0.6,
            'safety_car_probability': 0.4
        }
    
    def get_track_features(self, circuit_id: str) -> Dict[str, float]:
        """Get track-specific features for a circuit."""
        return self.track_characteristics.get(circuit_id, self.default_characteristics).copy()
    
    def calculate_track_suitability(self, circuit_id: str, driver_characteristics: Dict[str, float]) -> float:
        """
        Calculate how suitable a track is for a driver based on their characteristics.
        
        Args:
            circuit_id: Circuit identifier
            driver_characteristics: Driver's strengths/weaknesses
            
        Returns:
            Suitability score (0-1)
        """
        track_features = self.get_track_features(circuit_id)
        
        # Driver characteristics should include things like:
        # - overtaking_skill, wet_weather_skill, consistency, etc.
        suitability_factors = []
        
        # Overtaking skill vs track difficulty
        if 'overtaking_skill' in driver_characteristics:
            overtaking_factor = 1 - (track_features['overtaking_difficulty'] * (1 - driver_characteristics['overtaking_skill']))
            suitability_factors.append(overtaking_factor)
        
        # Weather skill vs weather sensitivity
        if 'wet_weather_skill' in driver_characteristics:
            weather_factor = 1 - (track_features['weather_sensitivity'] * (1 - driver_characteristics['wet_weather_skill']))
            suitability_factors.append(weather_factor)
        
        # Consistency vs safety car probability
        if 'consistency' in driver_characteristics:
            safety_car_factor = 1 - (track_features['safety_car_probability'] * (1 - driver_characteristics['consistency']))
            suitability_factors.append(safety_car_factor)
        
        return np.mean(suitability_factors) if suitability_factors else 0.5


class WeatherFeatures:
    """
    Weather-specific feature engineering.
    """
    
    def __init__(self):
        self.weather_impact_factors = {
            'temperature_optimal_range': (20, 30),  # Celsius
            'humidity_threshold': 70,  # Percentage
            'wind_speed_threshold': 15,  # km/h
            'pressure_normal_range': (1000, 1020)  # mbar
        }
    
    def extract_weather_features(self, weather: WeatherData, historical_weather: List[WeatherData] = None) -> Dict[str, float]:
        """Extract advanced weather features."""
        features = {}
        
        # Basic weather features
        features['temperature'] = weather.temperature
        features['humidity'] = weather.humidity
        features['pressure'] = weather.pressure
        features['wind_speed'] = weather.wind_speed
        features['wind_direction'] = weather.wind_direction
        features['track_temp'] = weather.track_temp
        features['rainfall'] = 1.0 if weather.rainfall else 0.0
        
        # Derived weather features
        features['temp_track_diff'] = weather.track_temp - weather.temperature
        features['temp_optimal'] = self._calculate_temperature_optimality(weather.temperature)
        features['humidity_high'] = 1.0 if weather.humidity > self.weather_impact_factors['humidity_threshold'] else 0.0
        features['wind_strong'] = 1.0 if weather.wind_speed > self.weather_impact_factors['wind_speed_threshold'] else 0.0
        features['pressure_normal'] = self._calculate_pressure_normality(weather.pressure)
        
        # Weather condition categories
        features['weather_dry'] = 1.0 if not weather.rainfall else 0.0
        features['weather_wet'] = 1.0 if weather.rainfall else 0.0
        features['weather_extreme'] = self._calculate_weather_extremity(weather)
        
        # Historical comparison if available
        if historical_weather:
            features.update(self._calculate_weather_comparison(weather, historical_weather))
        
        return features
    
    def _calculate_temperature_optimality(self, temperature: float) -> float:
        """Calculate how optimal the temperature is (0-1)."""
        optimal_min, optimal_max = self.weather_impact_factors['temperature_optimal_range']
        
        if optimal_min <= temperature <= optimal_max:
            return 1.0
        elif temperature < optimal_min:
            return max(0, 1 - (optimal_min - temperature) / 20)
        else:
            return max(0, 1 - (temperature - optimal_max) / 20)
    
    def _calculate_pressure_normality(self, pressure: float) -> float:
        """Calculate how normal the pressure is (0-1)."""
        normal_min, normal_max = self.weather_impact_factors['pressure_normal_range']
        
        if normal_min <= pressure <= normal_max:
            return 1.0
        else:
            deviation = min(abs(pressure - normal_min), abs(pressure - normal_max))
            return max(0, 1 - deviation / 50)
    
    def _calculate_weather_extremity(self, weather: WeatherData) -> float:
        """Calculate overall weather extremity (0-1)."""
        extremity_factors = []
        
        # Temperature extremity
        temp_extremity = 1 - self._calculate_temperature_optimality(weather.temperature)
        extremity_factors.append(temp_extremity)
        
        # Humidity extremity
        humidity_extremity = abs(weather.humidity - 50) / 50  # 50% is neutral
        extremity_factors.append(humidity_extremity)
        
        # Wind extremity
        wind_extremity = min(weather.wind_speed / 30, 1.0)  # 30 km/h is very windy
        extremity_factors.append(wind_extremity)
        
        # Rainfall
        if weather.rainfall:
            extremity_factors.append(1.0)
        
        return np.mean(extremity_factors)
    
    def _calculate_weather_comparison(self, current_weather: WeatherData, historical_weather: List[WeatherData]) -> Dict[str, float]:
        """Compare current weather to historical patterns."""
        if not historical_weather:
            return {}
        
        historical_temps = [w.temperature for w in historical_weather]
        historical_humidity = [w.humidity for w in historical_weather]
        historical_wind = [w.wind_speed for w in historical_weather]
        
        comparison_features = {}
        
        if historical_temps:
            comparison_features['temp_vs_avg'] = current_weather.temperature - np.mean(historical_temps)
            comparison_features['temp_percentile'] = np.percentile(historical_temps, 50)  # Simplified
        
        if historical_humidity:
            comparison_features['humidity_vs_avg'] = current_weather.humidity - np.mean(historical_humidity)
        
        if historical_wind:
            comparison_features['wind_vs_avg'] = current_weather.wind_speed - np.mean(historical_wind)
        
        return comparison_features


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering system that combines all feature extraction methods.
    """
    
    def __init__(self):
        self.base_extractor = BaseFeatureExtractor()
        self.rolling_calculator = RollingStatsCalculator()
        self.track_features = TrackSpecificFeatures()
        self.weather_features = WeatherFeatures()
        
        # Feature engineering configuration
        self.config = {
            'rolling_windows': [3, 5, 10],
            'interaction_features': True,
            'polynomial_features': True,
            'polynomial_degree': 2,
            'normalize_features': True,
            'binned_features': True,
            'outlier_detection': True
        }
    
    def engineer_race_features(self, target_race: RaceData, historical_races: List[RaceData],
                             include_target_drivers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Engineer comprehensive features for a race prediction.
        
        Args:
            target_race: The race to predict
            historical_races: Historical race data for feature engineering
            include_target_drivers: List of drivers to include (None for all)
            
        Returns:
            Dictionary with engineered features for each driver
        """
        try:
            logger.info(f"Engineering features for {target_race.race_name}")
            
            # Filter historical races (before target race)
            historical_races = [race for race in historical_races if race.date < target_race.date]
            
            if not historical_races:
                raise AdvancedFeatureEngineeringError("No historical data available for feature engineering")
            
            # Determine target drivers
            if include_target_drivers is None:
                target_drivers = [result.driver_id for result in target_race.results]
            else:
                target_drivers = include_target_drivers
            
            engineered_features = {}
            
            for driver_id in target_drivers:
                logger.debug(f"Engineering features for driver: {driver_id}")
                
                try:
                    driver_features = self._engineer_driver_features(
                        driver_id, target_race, historical_races
                    )
                    engineered_features[driver_id] = driver_features
                    
                except Exception as e:
                    logger.warning(f"Failed to engineer features for {driver_id}: {e}")
                    continue
            
            logger.info(f"Successfully engineered features for {len(engineered_features)} drivers")
            return engineered_features
            
        except Exception as e:
            logger.error(f"Error in race feature engineering: {e}")
            raise AdvancedFeatureEngineeringError(f"Failed to engineer race features: {e}")
    
    def _engineer_driver_features(self, driver_id: str, target_race: RaceData, 
                                historical_races: List[RaceData]) -> Dict[str, Any]:
        """Engineer comprehensive features for a single driver."""
        all_features = {}
        
        # 1. Base driver features
        try:
            driver_features = self.base_extractor.extract_driver_features(historical_races, driver_id)
            all_features.update({
                'recent_form': driver_features.recent_form,
                'constructor_performance': driver_features.constructor_performance,
                'track_experience': driver_features.track_experience,
                'weather_performance': driver_features.weather_performance,
                'qualifying_delta': driver_features.qualifying_delta,
                'championship_position': driver_features.championship_position,
                'points_total': driver_features.points_total
            })
        except Exception as e:
            logger.warning(f"Failed to extract base features for {driver_id}: {e}")
        
        # 2. Rolling statistics features
        for window in self.config['rolling_windows']:
            try:
                rolling_stats = self.rolling_calculator.calculate_driver_form(
                    historical_races, driver_id, window=window
                )
                
                # Extract latest rolling statistics
                if rolling_stats and rolling_stats['position_stats']:
                    latest_pos_stats = rolling_stats['position_stats'][-1]
                    all_features.update({
                        f'rolling_{window}_avg_position': latest_pos_stats.get('position_mean', 20),
                        f'rolling_{window}_position_std': latest_pos_stats.get('position_std', 0),
                        f'rolling_{window}_finish_rate': latest_pos_stats.get('finish_rate', 0)
                    })
                
                if rolling_stats and rolling_stats['points_stats']:
                    latest_points_stats = rolling_stats['points_stats'][-1]
                    all_features.update({
                        f'rolling_{window}_avg_points': latest_points_stats.get('points_per_race', 0),
                        f'rolling_{window}_points_std': latest_points_stats.get('points_std', 0)
                    })
                
                if rolling_stats and rolling_stats['form_trends']:
                    latest_trend = rolling_stats['form_trends'][-1]
                    all_features.update({
                        f'rolling_{window}_position_trend': latest_trend.get('position_trend', 0),
                        f'rolling_{window}_points_trend': latest_trend.get('points_trend', 0)
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to calculate rolling stats (window={window}) for {driver_id}: {e}")
        
        # 3. Track-specific features
        try:
            track_characteristics = self.track_features.get_track_features(target_race.circuit_id)
            all_features.update({f'track_{k}': v for k, v in track_characteristics.items()})
            
            # Track-specific performance
            track_performance = self.rolling_calculator.calculate_track_specific_form(
                historical_races, driver_id, 'driver', target_race.circuit_id
            )
            
            if track_performance:
                all_features.update({
                    'track_specific_avg_position': track_performance.get('avg_position', 20),
                    'track_specific_best_position': track_performance.get('best_position', 20),
                    'track_specific_finish_rate': track_performance.get('finish_rate', 0),
                    'track_specific_podium_rate': track_performance.get('podium_rate', 0),
                    'track_races_count': track_performance.get('races_at_track', 0)
                })
                
        except Exception as e:
            logger.warning(f"Failed to extract track features for {driver_id}: {e}")
        
        # 4. Weather features
        try:
            weather_features = self.weather_features.extract_weather_features(target_race.weather)
            all_features.update({f'weather_{k}': v for k, v in weather_features.items()})
        except Exception as e:
            logger.warning(f"Failed to extract weather features: {e}")
        
        # 5. Constructor features
        try:
            # Get driver's current constructor
            driver_constructor = None
            for result in target_race.results:
                if result.driver_id == driver_id:
                    driver_constructor = result.constructor_id
                    break
            
            if driver_constructor:
                constructor_performance = self.rolling_calculator.calculate_constructor_performance(
                    historical_races, driver_constructor, window=5
                )
                
                if constructor_performance and constructor_performance['performance_stats']:
                    latest_constructor_stats = constructor_performance['performance_stats'][-1]
                    all_features.update({
                        'constructor_avg_points': latest_constructor_stats.get('avg_points_per_race', 0),
                        'constructor_best_position': latest_constructor_stats.get('best_avg_position', 20),
                        'constructor_podium_races': latest_constructor_stats.get('podium_races', 0)
                    })
                
                if constructor_performance and constructor_performance['reliability_stats']:
                    latest_reliability = constructor_performance['reliability_stats'][-1]
                    all_features.update({
                        'constructor_reliability': latest_reliability.get('reliability_rate', 0),
                        'constructor_avg_finishers': latest_reliability.get('avg_finishers_per_race', 0)
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to extract constructor features for {driver_id}: {e}")
        
        # 6. Qualifying features (if available)
        try:
            driver_grid_position = None
            for result in target_race.results:
                if result.driver_id == driver_id:
                    driver_grid_position = result.grid_position
                    break
            
            if driver_grid_position:
                all_features.update({
                    'grid_position': driver_grid_position,
                    'grid_position_normalized': (21 - driver_grid_position) / 20,  # Higher is better
                    'starts_from_pole': 1.0 if driver_grid_position == 1 else 0.0,
                    'starts_from_front_row': 1.0 if driver_grid_position <= 2 else 0.0,
                    'starts_from_top_10': 1.0 if driver_grid_position <= 10 else 0.0
                })
                
        except Exception as e:
            logger.warning(f"Failed to extract qualifying features for {driver_id}: {e}")
        
        # 7. Apply advanced feature engineering techniques
        if self.config['interaction_features']:
            all_features = create_interaction_features(all_features)
        
        if self.config['polynomial_features']:
            all_features = create_polynomial_features(all_features, self.config['polynomial_degree'])
        
        if self.config['binned_features']:
            all_features = create_binned_features(all_features)
        
        if self.config['normalize_features']:
            all_features = normalize_features(all_features)
        
        return all_features
    
    def engineer_comparative_features(self, target_race: RaceData, historical_races: List[RaceData]) -> Dict[str, Any]:
        """
        Engineer comparative features between drivers for the target race.
        
        Args:
            target_race: The race to predict
            historical_races: Historical race data
            
        Returns:
            Dictionary with comparative features
        """
        try:
            comparative_features = {}
            
            # Get all drivers in the race
            drivers = [result.driver_id for result in target_race.results]
            
            if len(drivers) < 2:
                return comparative_features
            
            # Calculate head-to-head statistics for key driver pairs
            key_drivers = drivers[:10]  # Focus on top drivers to avoid too many combinations
            
            for i, driver1 in enumerate(key_drivers):
                for driver2 in key_drivers[i+1:]:
                    try:
                        h2h_stats = self.rolling_calculator.calculate_head_to_head_stats(
                            historical_races, driver1, driver2, window=10
                        )
                        
                        if h2h_stats and h2h_stats['overall_stats']:
                            overall = h2h_stats['overall_stats']
                            pair_key = f"{driver1}_vs_{driver2}"
                            
                            comparative_features[f"{pair_key}_total_races"] = overall.get('total_races', 0)
                            comparative_features[f"{pair_key}_driver1_win_rate"] = overall.get('driver1_win_percentage', 0) / 100
                            comparative_features[f"{pair_key}_points_advantage"] = overall.get('points_advantage_driver1', 0)
                            
                    except Exception as e:
                        logger.debug(f"Failed to calculate H2H for {driver1} vs {driver2}: {e}")
                        continue
            
            # Calculate field-relative features
            try:
                field_features = self._calculate_field_relative_features(target_race, historical_races)
                comparative_features.update(field_features)
            except Exception as e:
                logger.warning(f"Failed to calculate field-relative features: {e}")
            
            return comparative_features
            
        except Exception as e:
            logger.error(f"Error engineering comparative features: {e}")
            return {}
    
    def _calculate_field_relative_features(self, target_race: RaceData, historical_races: List[RaceData]) -> Dict[str, Any]:
        """Calculate features relative to the field strength."""
        field_features = {}
        
        try:
            # Calculate average field strength metrics
            all_drivers = set()
            for race in historical_races[-10:]:  # Last 10 races
                for result in race.results:
                    all_drivers.add(result.driver_id)
            
            # Field strength indicators
            field_features['field_size'] = len(all_drivers)
            field_features['championship_contenders'] = min(len(all_drivers), 8)  # Estimate
            
            # Constructor diversity
            constructors = set()
            for result in target_race.results:
                constructors.add(result.constructor_id)
            
            field_features['constructor_diversity'] = len(constructors)
            field_features['competitive_balance'] = min(len(constructors) / 10, 1.0)  # Normalized
            
        except Exception as e:
            logger.warning(f"Error calculating field features: {e}")
        
        return field_features
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get feature importance weights for different feature categories."""
        return {
            'recent_form': 0.25,
            'rolling_stats': 0.20,
            'track_specific': 0.15,
            'constructor': 0.15,
            'qualifying': 0.10,
            'weather': 0.08,
            'comparative': 0.05,
            'interaction': 0.02
        }
    
    def validate_engineered_features(self, features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate engineered features for quality and completeness.
        
        Args:
            features: Dictionary of driver features
            
        Returns:
            Validation report
        """
        validation_report = {
            'total_drivers': len(features),
            'feature_counts': {},
            'missing_features': [],
            'outlier_features': [],
            'quality_score': 0.0
        }
        
        if not features:
            validation_report['quality_score'] = 0.0
            return validation_report
        
        # Analyze feature completeness
        all_feature_names = set()
        for driver_features in features.values():
            all_feature_names.update(driver_features.keys())
        
        validation_report['total_features'] = len(all_feature_names)
        
        # Check feature coverage for each driver
        feature_coverage = {}
        for feature_name in all_feature_names:
            coverage_count = sum(1 for driver_features in features.values() 
                               if feature_name in driver_features and driver_features[feature_name] is not None)
            feature_coverage[feature_name] = coverage_count / len(features)
        
        # Identify missing features (< 80% coverage)
        validation_report['missing_features'] = [
            feature for feature, coverage in feature_coverage.items() 
            if coverage < 0.8
        ]
        
        # Calculate quality score
        avg_coverage = np.mean(list(feature_coverage.values()))
        validation_report['quality_score'] = avg_coverage
        
        # Feature category analysis
        feature_categories = {
            'rolling': [f for f in all_feature_names if 'rolling' in f],
            'track': [f for f in all_feature_names if 'track' in f],
            'weather': [f for f in all_feature_names if 'weather' in f],
            'constructor': [f for f in all_feature_names if 'constructor' in f],
            'interaction': [f for f in all_feature_names if 'interaction' in f]
        }
        
        validation_report['feature_counts'] = {
            category: len(features) for category, features in feature_categories.items()
        }
        
        return validation_report