"""
Base feature extraction for F1 race prediction.
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ..models.data_models import RaceData, RaceResult, QualifyingResult, DriverFeatures
from ..models.interfaces import FeatureEngineer
from ..config import config


logger = logging.getLogger(__name__)


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class BaseFeatureExtractor(FeatureEngineer):
    """
    Base feature extractor that converts raw F1 data into ML features.
    """
    
    def __init__(self):
        self.min_races_for_features = config.data.min_races_for_features
        self.feature_cache = {}
        
        # Define feature categories
        self.driver_features = [
            'recent_form', 'championship_position', 'points_total',
            'avg_grid_position', 'avg_finish_position', 'finish_rate',
            'points_per_race', 'podium_rate', 'dnf_rate'
        ]
        
        self.constructor_features = [
            'constructor_performance', 'constructor_reliability',
            'constructor_points_per_race', 'constructor_podium_rate'
        ]
        
        self.qualifying_features = [
            'qualifying_delta', 'q1_time_normalized', 'q2_time_normalized',
            'q3_time_normalized', 'qualifying_position'
        ]
        
        self.track_features = [
            'track_experience', 'track_performance_history',
            'track_specific_form'
        ]
        
        self.weather_features = [
            'weather_performance', 'rain_performance', 'temperature_factor'
        ]
    
    def extract_driver_features(self, race_data: List[RaceData], driver_id: str) -> DriverFeatures:
        """Extract comprehensive features for a specific driver."""
        try:
            if not race_data:
                raise FeatureExtractionError("No race data provided", {"driver_id": driver_id})
            
            # Sort races by date
            sorted_races = sorted(race_data, key=lambda x: x.date)
            
            # Find driver's results in all races
            driver_results = []
            driver_qualifying = []
            
            for race in sorted_races:
                # Find driver in race results
                for result in race.results:
                    if result.driver_id == driver_id:
                        driver_results.append((race, result))
                        break
                
                # Find driver in qualifying
                for qual in race.qualifying:
                    if qual.driver_id == driver_id:
                        driver_qualifying.append((race, qual))
                        break
            
            if not driver_results:
                raise FeatureExtractionError(f"No results found for driver {driver_id}")
            
            # Extract base features
            features = self._extract_base_driver_features(driver_results, driver_qualifying)
            
            # Calculate recent form
            features['recent_form'] = self._calculate_recent_form(driver_results)
            
            # Calculate constructor performance
            features['constructor_performance'] = self._calculate_constructor_performance(driver_results)
            
            # Calculate qualifying performance
            features['qualifying_delta'] = self._calculate_qualifying_delta(driver_qualifying)
            
            # Calculate track experience
            features['track_experience'] = len(driver_results)
            
            # Calculate weather performance (placeholder - would need weather correlation)
            features['weather_performance'] = 0.5  # Neutral baseline
            
            # Get latest constructor and championship info
            latest_result = driver_results[-1][1]
            features['constructor_id'] = latest_result.constructor_id
            
            return DriverFeatures(
                driver_id=driver_id,
                recent_form=features['recent_form'],
                constructor_performance=features['constructor_performance'],
                track_experience=features['track_experience'],
                weather_performance=features['weather_performance'],
                qualifying_delta=features['qualifying_delta'],
                championship_position=features.get('championship_position', 20),
                points_total=features.get('points_total', 0)
            )
            
        except Exception as e:
            logger.error(f"Error extracting features for driver {driver_id}: {e}")
            raise FeatureExtractionError(f"Failed to extract driver features: {e}", {"driver_id": driver_id})
    
    def _extract_base_driver_features(self, driver_results: List[Tuple[RaceData, RaceResult]], 
                                    driver_qualifying: List[Tuple[RaceData, QualifyingResult]]) -> Dict[str, float]:
        """Extract base statistical features for a driver."""
        features = {}
        
        if not driver_results:
            return features
        
        # Basic statistics
        positions = [result.final_position for _, result in driver_results if result.final_position < 999]
        grid_positions = [result.grid_position for _, result in driver_results]
        points = [result.points for _, result in driver_results]
        
        # Position-based features
        if positions:
            features['avg_finish_position'] = np.mean(positions)
            features['median_finish_position'] = np.median(positions)
            features['best_finish'] = min(positions)
            features['worst_finish'] = max(positions)
            features['finish_consistency'] = np.std(positions) if len(positions) > 1 else 0
        else:
            features['avg_finish_position'] = 20
            features['median_finish_position'] = 20
            features['best_finish'] = 20
            features['worst_finish'] = 20
            features['finish_consistency'] = 0
        
        # Grid position features
        if grid_positions:
            features['avg_grid_position'] = np.mean(grid_positions)
            features['grid_consistency'] = np.std(grid_positions) if len(grid_positions) > 1 else 0
        else:
            features['avg_grid_position'] = 20
            features['grid_consistency'] = 0
        
        # Points features
        features['points_total'] = sum(points)
        features['points_per_race'] = np.mean(points) if points else 0
        features['points_consistency'] = np.std(points) if len(points) > 1 else 0
        
        # Rate-based features
        total_races = len(driver_results)
        finished_races = len(positions)
        podium_finishes = len([p for p in positions if p <= 3])
        wins = len([p for p in positions if p == 1])
        
        features['finish_rate'] = finished_races / total_races if total_races > 0 else 0
        features['podium_rate'] = podium_finishes / total_races if total_races > 0 else 0
        features['win_rate'] = wins / total_races if total_races > 0 else 0
        features['dnf_rate'] = 1 - features['finish_rate']
        
        # Performance improvement
        if len(positions) >= 2:
            recent_positions = positions[-3:] if len(positions) >= 3 else positions[-2:]
            early_positions = positions[:3] if len(positions) >= 3 else positions[:2]
            features['performance_trend'] = np.mean(early_positions) - np.mean(recent_positions)
        else:
            features['performance_trend'] = 0
        
        # Championship position (estimated based on points)
        features['championship_position'] = self._estimate_championship_position(features['points_total'])
        
        return features
    
    def _calculate_recent_form(self, driver_results: List[Tuple[RaceData, RaceResult]], window: int = 5) -> float:
        """Calculate driver's recent form based on last N races."""
        if not driver_results:
            return 0.5  # Neutral form
        
        # Get recent results
        recent_results = driver_results[-window:] if len(driver_results) >= window else driver_results
        
        if not recent_results:
            return 0.5
        
        # Calculate form based on finishing positions and points
        form_scores = []
        for _, result in recent_results:
            if result.final_position < 999:  # Finished the race
                # Convert position to score (1st = 1.0, 20th = 0.0)
                position_score = max(0, (21 - result.final_position) / 20)
                
                # Add points bonus
                points_score = min(result.points / 25, 1.0)  # Normalize to max 25 points
                
                # Combine scores
                race_score = (position_score * 0.7) + (points_score * 0.3)
                form_scores.append(race_score)
            else:
                # DNF gets low score
                form_scores.append(0.1)
        
        if not form_scores:
            return 0.5
        
        # Weight recent races more heavily
        weights = np.linspace(0.5, 1.0, len(form_scores))
        weighted_form = np.average(form_scores, weights=weights)
        
        return float(weighted_form)
    
    def _calculate_constructor_performance(self, driver_results: List[Tuple[RaceData, RaceResult]]) -> float:
        """Calculate constructor performance based on driver's results."""
        if not driver_results:
            return 0.5
        
        # Get constructor from most recent race
        latest_constructor = driver_results[-1][1].constructor_id
        
        # Calculate constructor performance based on this driver's results
        constructor_scores = []
        for _, result in driver_results:
            if result.constructor_id == latest_constructor:
                if result.final_position < 999:
                    # Convert position to score
                    score = max(0, (21 - result.final_position) / 20)
                    constructor_scores.append(score)
                else:
                    constructor_scores.append(0.1)
        
        if not constructor_scores:
            return 0.5
        
        return float(np.mean(constructor_scores))
    
    def _calculate_qualifying_delta(self, driver_qualifying: List[Tuple[RaceData, QualifyingResult]]) -> float:
        """Calculate qualifying performance delta from expected position."""
        if not driver_qualifying:
            return 0.0
        
        # Calculate average qualifying position
        qualifying_positions = []
        for _, qual in driver_qualifying:
            if qual.position:
                qualifying_positions.append(qual.position)
        
        if not qualifying_positions:
            return 0.0
        
        avg_qualifying = np.mean(qualifying_positions)
        expected_position = 10.5  # Middle of the grid
        
        # Positive delta means better than expected
        delta = expected_position - avg_qualifying
        
        # Normalize to [-1, 1] range
        normalized_delta = max(-1, min(1, delta / 10))
        
        return float(normalized_delta)
    
    def _estimate_championship_position(self, points_total: float) -> int:
        """Estimate championship position based on points total."""
        # This is a rough estimation - in practice, you'd calculate from actual standings
        if points_total >= 300:
            return 1
        elif points_total >= 250:
            return 2
        elif points_total >= 200:
            return 3
        elif points_total >= 150:
            return 4
        elif points_total >= 100:
            return 5
        elif points_total >= 80:
            return 6
        elif points_total >= 60:
            return 7
        elif points_total >= 40:
            return 8
        elif points_total >= 20:
            return 9
        elif points_total >= 10:
            return 10
        else:
            return min(20, max(11, int(21 - points_total)))
    
    def calculate_rolling_stats(self, races: List[RaceData], window: int) -> Dict[str, Any]:
        """Calculate rolling statistics over a window of races."""
        try:
            if not races or window <= 0:
                return {}
            
            # Sort races by date
            sorted_races = sorted(races, key=lambda x: x.date)
            
            rolling_stats = {
                'driver_form': {},
                'constructor_performance': {},
                'track_statistics': {},
                'weather_patterns': {}
            }
            
            # Calculate rolling statistics for each driver
            all_drivers = set()
            for race in sorted_races:
                for result in race.results:
                    all_drivers.add(result.driver_id)
            
            for driver_id in all_drivers:
                driver_rolling_form = []
                
                for i in range(len(sorted_races)):
                    # Get window of races ending at position i
                    start_idx = max(0, i - window + 1)
                    window_races = sorted_races[start_idx:i + 1]
                    
                    # Calculate form for this window
                    driver_results = []
                    for race in window_races:
                        for result in race.results:
                            if result.driver_id == driver_id:
                                driver_results.append((race, result))
                                break
                    
                    if driver_results:
                        form = self._calculate_recent_form(driver_results, len(driver_results))
                        driver_rolling_form.append({
                            'race_date': sorted_races[i].date,
                            'form': form,
                            'races_in_window': len(driver_results)
                        })
                
                rolling_stats['driver_form'][driver_id] = driver_rolling_form
            
            # Calculate constructor rolling performance
            all_constructors = set()
            for race in sorted_races:
                for result in race.results:
                    all_constructors.add(result.constructor_id)
            
            for constructor_id in all_constructors:
                constructor_rolling_perf = []
                
                for i in range(len(sorted_races)):
                    start_idx = max(0, i - window + 1)
                    window_races = sorted_races[start_idx:i + 1]
                    
                    # Calculate constructor performance
                    constructor_results = []
                    for race in window_races:
                        for result in race.results:
                            if result.constructor_id == constructor_id:
                                constructor_results.append(result)
                    
                    if constructor_results:
                        avg_position = np.mean([r.final_position for r in constructor_results if r.final_position < 999])
                        performance = max(0, (21 - avg_position) / 20) if avg_position < 999 else 0.1
                        
                        constructor_rolling_perf.append({
                            'race_date': sorted_races[i].date,
                            'performance': performance,
                            'results_count': len(constructor_results)
                        })
                
                rolling_stats['constructor_performance'][constructor_id] = constructor_rolling_perf
            
            return rolling_stats
            
        except Exception as e:
            logger.error(f"Error calculating rolling stats: {e}")
            raise FeatureExtractionError(f"Failed to calculate rolling stats: {e}")
    
    def encode_categorical_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical variables for ML models."""
        try:
            encoded_features = features.copy()
            
            # Define categorical mappings
            categorical_mappings = {
                'constructor_id': self._get_constructor_encoding(),
                'circuit_id': self._get_circuit_encoding(),
                'weather_condition': self._get_weather_encoding()
            }
            
            for feature_name, value in features.items():
                if feature_name in categorical_mappings:
                    mapping = categorical_mappings[feature_name]
                    
                    if isinstance(value, str):
                        # One-hot encoding
                        for category in mapping:
                            encoded_features[f"{feature_name}_{category}"] = 1.0 if value == category else 0.0
                        
                        # Remove original categorical feature
                        del encoded_features[feature_name]
                    
                    elif isinstance(value, list):
                        # Handle list of categorical values
                        for category in mapping:
                            encoded_features[f"{feature_name}_{category}"] = 1.0 if category in value else 0.0
                        
                        del encoded_features[feature_name]
            
            return encoded_features
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            raise FeatureExtractionError(f"Failed to encode categorical features: {e}")
    
    def _get_constructor_encoding(self) -> List[str]:
        """Get list of constructor IDs for encoding."""
        return [
            'mercedes', 'red_bull', 'ferrari', 'mclaren', 'alpine',
            'aston_martin', 'williams', 'alphatauri', 'alfa', 'haas'
        ]
    
    def _get_circuit_encoding(self) -> List[str]:
        """Get list of circuit IDs for encoding."""
        return [
            'bahrain', 'saudi_arabia', 'australia', 'imola', 'miami',
            'spain', 'monaco', 'azerbaijan', 'canada', 'britain',
            'austria', 'france', 'hungary', 'belgium', 'netherlands',
            'italy', 'singapore', 'japan', 'qatar', 'usa', 'mexico',
            'brazil', 'abu_dhabi'
        ]
    
    def _get_weather_encoding(self) -> List[str]:
        """Get list of weather conditions for encoding."""
        return ['dry', 'wet', 'mixed', 'unknown']
    
    def extract_race_features(self, race_data: RaceData, historical_data: List[RaceData] = None) -> Dict[str, Any]:
        """Extract features for a complete race."""
        try:
            race_features = {
                'race_metadata': self._extract_race_metadata_features(race_data),
                'weather_features': self._extract_weather_features(race_data),
                'grid_features': self._extract_grid_features(race_data),
                'driver_features': {},
                'constructor_features': {}
            }
            
            # Extract features for each driver
            for result in race_data.results:
                driver_id = result.driver_id
                
                # Get historical data for this driver
                if historical_data:
                    driver_history = [race for race in historical_data if race.date < race_data.date]
                    if driver_history:
                        driver_features = self.extract_driver_features(driver_history, driver_id)
                        race_features['driver_features'][driver_id] = {
                            'recent_form': driver_features.recent_form,
                            'constructor_performance': driver_features.constructor_performance,
                            'track_experience': driver_features.track_experience,
                            'weather_performance': driver_features.weather_performance,
                            'qualifying_delta': driver_features.qualifying_delta,
                            'championship_position': driver_features.championship_position,
                            'points_total': driver_features.points_total
                        }
                
                # Add current race specific features
                race_features['driver_features'][driver_id] = race_features['driver_features'].get(driver_id, {})
                race_features['driver_features'][driver_id].update({
                    'grid_position': result.grid_position,
                    'constructor_id': result.constructor_id
                })
            
            return race_features
            
        except Exception as e:
            logger.error(f"Error extracting race features: {e}")
            raise FeatureExtractionError(f"Failed to extract race features: {e}")
    
    def _extract_race_metadata_features(self, race_data: RaceData) -> Dict[str, Any]:
        """Extract metadata features from race data."""
        return {
            'season': race_data.season,
            'round': race_data.round,
            'circuit_id': race_data.circuit_id,
            'race_date': race_data.date.isoformat(),
            'day_of_year': race_data.date.timetuple().tm_yday,
            'month': race_data.date.month,
            'is_weekend': race_data.date.weekday() >= 5
        }
    
    def _extract_weather_features(self, race_data: RaceData) -> Dict[str, Any]:
        """Extract weather-related features."""
        weather = race_data.weather
        return {
            'temperature': weather.temperature,
            'humidity': weather.humidity,
            'pressure': weather.pressure,
            'wind_speed': weather.wind_speed,
            'wind_direction': weather.wind_direction,
            'rainfall': 1.0 if weather.rainfall else 0.0,
            'track_temp': weather.track_temp,
            'temp_track_diff': weather.track_temp - weather.temperature,
            'weather_condition': 'wet' if weather.rainfall else 'dry'
        }
    
    def _extract_grid_features(self, race_data: RaceData) -> Dict[str, Any]:
        """Extract grid-related features."""
        if not race_data.qualifying:
            return {}
        
        grid_positions = [q.position for q in race_data.qualifying]
        
        return {
            'grid_spread': max(grid_positions) - min(grid_positions) if grid_positions else 0,
            'avg_grid_position': np.mean(grid_positions) if grid_positions else 10.5,
            'grid_competitiveness': np.std(grid_positions) if len(grid_positions) > 1 else 0
        }