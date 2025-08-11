"""
Core prediction engine for F1 race predictions.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from ..models.implementations import ModelFactory
from ..models.ensemble import AdvancedEnsembleModel
from ..models.data_models import RaceParameters, RacePrediction, PositionPrediction
from ..features.base_extractor import BaseFeatureExtractor
from ..features.rolling_stats import RollingStatsCalculator
from ..features.advanced_features import BaseFeatureExtractor as AdvancedFeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class PredictionRequest:
    """Request structure for race predictions."""
    race_name: str
    circuit: str
    date: datetime
    drivers: List[Dict[str, Any]]
    weather: Dict[str, Any]
    session_type: str = "race"  # race, qualifying, sprint
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['date'] = self.date.isoformat()
        return data


@dataclass
class PredictionResult:
    """Result structure for race predictions."""
    race_name: str
    predictions: List[PositionPrediction]
    confidence_score: float
    prediction_metadata: Dict[str, Any]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'race_name': self.race_name,
            'predictions': [pred.to_dict() for pred in self.predictions],
            'confidence_score': self.confidence_score,
            'prediction_metadata': self.prediction_metadata,
            'generated_at': self.generated_at.isoformat()
        }


class ConfidenceCalculator:
    """
    Calculate prediction confidence and uncertainty quantification.
    """
    
    def __init__(self):
        self.confidence_factors = {
            'model_agreement': 0.3,
            'prediction_variance': 0.25,
            'historical_accuracy': 0.25,
            'data_quality': 0.2
        }
    
    def calculate_prediction_confidence(self, predictions: List[np.ndarray], 
                                      model_accuracies: List[float] = None,
                                      data_quality_score: float = 1.0) -> Dict[str, float]:
        """
        Calculate overall confidence for predictions.
        
        Args:
            predictions: List of prediction arrays from different models
            model_accuracies: Historical accuracy scores for each model
            data_quality_score: Quality score of input data (0-1)
            
        Returns:
            Dictionary with confidence metrics
        """
        if not predictions:
            return {'overall_confidence': 0.0}
        
        confidence_metrics = {}
        
        # Model agreement confidence
        if len(predictions) > 1:
            agreement_score = self._calculate_model_agreement(predictions)
            confidence_metrics['model_agreement'] = agreement_score
        else:
            confidence_metrics['model_agreement'] = 0.5  # Neutral for single model
        
        # Prediction variance confidence
        variance_score = self._calculate_variance_confidence(predictions)
        confidence_metrics['prediction_variance'] = variance_score
        
        # Historical accuracy confidence
        if model_accuracies:
            accuracy_score = np.mean(model_accuracies)
            confidence_metrics['historical_accuracy'] = accuracy_score
        else:
            confidence_metrics['historical_accuracy'] = 0.7  # Default assumption
        
        # Data quality confidence
        confidence_metrics['data_quality'] = data_quality_score
        
        # Calculate overall confidence
        overall_confidence = sum(
            confidence_metrics[factor] * weight 
            for factor, weight in self.confidence_factors.items()
            if factor in confidence_metrics
        )
        
        confidence_metrics['overall_confidence'] = overall_confidence
        
        return confidence_metrics
    
    def _calculate_model_agreement(self, predictions: List[np.ndarray]) -> float:
        """Calculate agreement between different model predictions."""
        if len(predictions) < 2:
            return 1.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if not correlations:
            return 0.5
        
        # Average correlation as agreement score
        return np.mean(correlations)
    
    def _calculate_variance_confidence(self, predictions: List[np.ndarray]) -> float:
        """Calculate confidence based on prediction variance."""
        if len(predictions) == 1:
            # For single model, use internal variance
            pred_var = np.var(predictions[0])
            # Lower variance = higher confidence
            return max(0.0, 1.0 - pred_var / 100.0)  # Normalize by expected max variance
        
        # For multiple models, calculate variance across models
        pred_array = np.array(predictions)
        mean_variance = np.mean(np.var(pred_array, axis=0))
        
        # Convert variance to confidence (lower variance = higher confidence)
        return max(0.0, 1.0 - mean_variance / 50.0)  # Normalize
    
    def calculate_position_confidence(self, position_predictions: List[float]) -> float:
        """Calculate confidence for a specific position prediction."""
        if not position_predictions:
            return 0.0
        
        # Calculate standard deviation of predictions
        std_dev = np.std(position_predictions)
        
        # Convert to confidence (lower std = higher confidence)
        # Assuming max reasonable std is 5 positions
        confidence = max(0.0, 1.0 - std_dev / 5.0)
        
        return confidence


class PredictionEngine:
    """
    Core prediction engine for generating F1 race predictions.
    """
    
    def __init__(self, model_type: str = 'ensemble', model_config: Dict[str, Any] = None):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = None
        self.feature_extractor = BaseFeatureExtractor()
        self.rolling_calculator = RollingStatsCalculator()
        self.advanced_extractor = AdvancedFeatureExtractor()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Prediction cache for performance
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour in seconds
        
        # Model performance tracking
        self.model_accuracies = {}
        
    def initialize_model(self, training_data: List[Dict[str, Any]] = None,
                        training_targets: List[int] = None) -> bool:
        """
        Initialize and train the prediction model.
        
        Args:
            training_data: Historical race data for training
            training_targets: Historical race results
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info(f"Initializing prediction model: {self.model_type}")
            
            if self.model_type == 'ensemble':
                self.model = AdvancedEnsembleModel(
                    ensemble_method='weighted_average',
                    **self.model_config
                )
            else:
                self.model = ModelFactory.create_model(
                    self.model_type, 
                    **self.model_config
                )
            
            # Train model if training data provided
            if training_data and training_targets:
                logger.info("Training model with provided data")
                training_results = self.model.train(
                    training_data, training_targets, 
                    tune_hyperparameters=False
                )
                
                # Store model accuracy for confidence calculation
                if 'metrics' in training_results:
                    mae = training_results['metrics'].get('mae', float('inf'))
                    # Convert MAE to accuracy score (lower MAE = higher accuracy)
                    accuracy = max(0.0, 1.0 - mae / 10.0)  # Normalize
                    self.model_accuracies[self.model_type] = accuracy
                
                logger.info("Model training completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def predict_race(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate predictions for a single race.
        
        Args:
            request: Race prediction request
            
        Returns:
            Race prediction result
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        logger.info(f"Generating predictions for race: {request.race_name}")
        
        # Disable caching for more varied predictions
        # cache_key = self._generate_cache_key(request)
        # cached_result = self._get_cached_prediction(cache_key)
        # if cached_result:
        #     logger.info("Returning cached prediction")
        #     return cached_result
        
        try:
            # Extract features for all drivers
            driver_features = self._extract_race_features(request)
            
            # Make predictions with race day variability
            base_predictions = self.model.predict(driver_features)
            
            # Add significant randomness to base predictions for variety
            import random
            import numpy as np
            
            # Shuffle predictions occasionally for more variety
            if random.random() < 0.4:  # 40% chance of major shuffle
                indices = list(range(len(base_predictions)))
                random.shuffle(indices)
                base_predictions = [base_predictions[i] for i in indices]
            
            # Add race day chaos and unpredictability
            race_chaos = random.uniform(0.6, 1.4)  # Increased unpredictability range
            
            # Create position predictions
            position_predictions = []
            for i, driver in enumerate(request.drivers):
                # Add individual driver variability
                driver_variability = random.uniform(-1.5, 1.5) * race_chaos
                adjusted_prediction = base_predictions[i] + driver_variability
                
                pred_position = max(1, min(len(request.drivers), round(adjusted_prediction)))
                
                position_pred = PositionPrediction(
                    driver_id=driver.get('driver_id', f"driver_{i}"),
                    predicted_position=int(pred_position),
                    probability_distribution=list(self._calculate_position_probabilities(adjusted_prediction).values()),
                    expected_points=self._calculate_expected_points(pred_position),
                    confidence_score=max(0.1, min(0.95, 
                        self.confidence_calculator.calculate_position_confidence([base_predictions[i]]) + 
                        random.uniform(-0.1, 0.1)))
                )
                # Add driver name as additional attribute
                position_pred.driver_name = driver.get('name', f"Driver {i+1}")
                position_predictions.append(position_pred)
            
            # Sort by predicted position and ensure unique positions
            position_predictions.sort(key=lambda x: x.predicted_position)
            
            # Ensure unique positions (resolve ties)
            used_positions = set()
            for i, pred in enumerate(position_predictions):
                original_pos = pred.predicted_position
                
                # If position is already used, find next available
                while pred.predicted_position in used_positions:
                    pred.predicted_position += 1
                    if pred.predicted_position > len(request.drivers):
                        pred.predicted_position = len(request.drivers)
                        break
                
                used_positions.add(pred.predicted_position)
            
            # Sort again after position adjustment
            position_predictions.sort(key=lambda x: x.predicted_position)
            
            # Calculate overall confidence with variability
            base_confidence_metrics = self.confidence_calculator.calculate_prediction_confidence(
                [base_predictions],
                list(self.model_accuracies.values()) if self.model_accuracies else None
            )
            
            # Add confidence variability
            confidence_variation = random.uniform(-0.05, 0.05)
            overall_confidence = max(0.1, min(0.95, 
                base_confidence_metrics['overall_confidence'] + confidence_variation))
            
            confidence_metrics = base_confidence_metrics.copy()
            confidence_metrics['overall_confidence'] = overall_confidence
            
            # Create result
            result = PredictionResult(
                race_name=request.race_name,
                predictions=position_predictions,
                confidence_score=confidence_metrics['overall_confidence'],
                prediction_metadata={
                    'model_type': self.model_type,
                    'confidence_breakdown': confidence_metrics,
                    'feature_count': len(driver_features[0]) if driver_features else 0,
                    'driver_count': len(request.drivers)
                },
                generated_at=datetime.now()
            )
            
            # Disable caching for varied predictions
            # self._cache_prediction(cache_key, result)
            
            logger.info(f"Predictions generated successfully. Overall confidence: {result.confidence_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """
        Generate predictions for multiple races.
        
        Args:
            requests: List of race prediction requests
            
        Returns:
            List of race prediction results
        """
        logger.info(f"Generating batch predictions for {len(requests)} races")
        
        results = []
        for request in requests:
            try:
                result = self.predict_race(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict race {request.race_name}: {e}")
                # Create error result
                error_result = PredictionResult(
                    race_name=request.race_name,
                    predictions=[],
                    confidence_score=0.0,
                    prediction_metadata={'error': str(e)},
                    generated_at=datetime.now()
                )
                results.append(error_result)
        
        logger.info(f"Batch prediction completed. {len(results)} results generated")
        return results
    
    def _extract_race_features(self, request: PredictionRequest) -> List[Dict[str, Any]]:
        """Extract features for all drivers in the race using realistic F1 data format."""
        from web.utils.realistic_f1_data import RealisticF1Data
        
        f1_data = RealisticF1Data()
        driver_features = []
        circuit_data = f1_data.get_circuit_data(request.circuit)
        
        for i, driver in enumerate(request.drivers):
            driver_id = driver.get('driver_id', '')
            driver_data = f1_data.get_driver_data(driver_id)
            
            if not driver_data:
                # Fallback for unknown drivers
                driver_data = {
                    'points': 0, 'wins': 0, 'team': 'Unknown',
                    'skill_rating': 0.70, 'consistency': 0.70, 'wet_weather_skill': 0.70
                }
            
            team_data = f1_data.get_team_data(driver_data.get('team', 'Unknown'))
            weather = request.weather
            is_wet = weather.get('conditions', 'dry') != 'dry'
            
            # Create feature vector matching the training data format (24 features)
            features = {
                'qualifying_position': driver.get('grid_position', i + 1),
                'driver_championship_points': driver_data.get('points', 0),
                'constructor_championship_points': int(team_data.get('car_rating', 0.6) * 600),
                'driver_wins_season': driver_data.get('wins', 0),
                'constructor_wins_season': max(0, driver_data.get('wins', 0)),
                'track_temperature': weather.get('track_temp', 25.0),
                'air_temperature': weather.get('air_temp', 20.0),
                'humidity': weather.get('humidity', 60.0),
                'wind_speed': weather.get('wind_speed', 5.0),
                'weather_dry': 1 if not is_wet else 0,
                'track_grip': weather.get('grip_level', 0.9 if not is_wet else 0.6),
                'fuel_load': driver.get('fuel_load', 100.0),
                'tire_compound': driver.get('tire_compound', 2),
                'driver_experience': min(400, driver_data.get('points', 0) + 100),
                'car_performance_rating': team_data.get('car_rating', 0.6),
                'engine_power': driver.get('engine_power', 900.0),
                'aerodynamic_efficiency': team_data.get('car_rating', 0.6) * 0.95,
                'driver_skill_rating': driver_data.get('skill_rating', 0.70),
                'consistency_rating': driver_data.get('consistency', 0.70),
                'wet_weather_skill': driver_data.get('wet_weather_skill', 0.70),
                'reliability_rating': team_data.get('reliability', 0.75),
                'strategy_rating': team_data.get('strategy', 0.70),
                'circuit_overtaking_factor': circuit_data.get('overtaking', 0.6),
                'qualifying_importance': circuit_data.get('qualifying_importance', 0.65)
            }
            
            driver_features.append(features)
        
        return driver_features
    
    def _calculate_expected_points(self, predicted_position: float) -> float:
        """Calculate expected championship points based on predicted position."""
        # F1 2024 points system
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # If predicted position is in points, return those points
        if 1 <= predicted_position <= 10:
            # Interpolate between positions for fractional predictions
            lower_pos = int(predicted_position)
            upper_pos = min(10, lower_pos + 1)
            
            lower_points = points_system.get(lower_pos, 0)
            upper_points = points_system.get(upper_pos, 0)
            
            # Linear interpolation
            fraction = predicted_position - lower_pos
            expected_points = lower_points * (1 - fraction) + upper_points * fraction
            
            return expected_points
        
        return 0.0  # No points for positions > 10
    
    def _calculate_position_probabilities(self, predicted_position: float) -> Dict[int, float]:
        """Calculate probability distribution for finishing positions."""
        probabilities = {}
        
        # Use normal distribution around predicted position
        std_dev = 2.0  # Standard deviation for position uncertainty
        
        for position in range(1, 21):  # F1 has max 20 cars
            # Calculate probability using normal distribution
            prob = np.exp(-0.5 * ((position - predicted_position) / std_dev) ** 2)
            probabilities[position] = prob
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {pos: prob / total_prob for pos, prob in probabilities.items()}
        
        return probabilities
    
    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction request."""
        # Create a hash of key request parameters
        key_data = {
            'race_name': request.race_name,
            'circuit': request.circuit,
            'date': request.date.isoformat(),
            'drivers': sorted([d.get('driver_id', str(i)) for i, d in enumerate(request.drivers)]),
            'weather': request.weather,
            'model_type': self.model_type
        }
        
        import hashlib
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResult]:
        """Get cached prediction if available and not expired."""
        if cache_key not in self.prediction_cache:
            return None
        
        cached_data = self.prediction_cache[cache_key]
        
        # Check if cache is expired
        cache_time = cached_data['timestamp']
        if (datetime.now() - cache_time).total_seconds() > self.cache_ttl:
            del self.prediction_cache[cache_key]
            return None
        
        return cached_data['result']
    
    def _cache_prediction(self, cache_key: str, result: PredictionResult):
        """Cache prediction result."""
        self.prediction_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries if cache gets too large
        if len(self.prediction_cache) > 100:
            self._clean_cache()
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, data in self.prediction_cache.items():
            if (current_time - data['timestamp']).total_seconds() > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.prediction_cache[key]
        
        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def get_prediction_insights(self, result: PredictionResult) -> Dict[str, Any]:
        """
        Generate insights from prediction results.
        
        Args:
            result: Prediction result to analyze
            
        Returns:
            Dictionary with prediction insights
        """
        insights = {
            'race_name': result.race_name,
            'total_drivers': len(result.predictions),
            'confidence_level': 'high' if result.confidence_score > 0.8 else 'medium' if result.confidence_score > 0.6 else 'low',
            'most_likely_winner': None,
            'biggest_surprises': [],
            'closest_battles': [],
            'confidence_distribution': {}
        }
        
        if not result.predictions:
            return insights
        
        # Most likely winner
        winner = min(result.predictions, key=lambda x: x.predicted_position)
        insights['most_likely_winner'] = {
            'driver': getattr(winner, 'driver_name', winner.driver_id),
            'confidence': winner.confidence_score
        }
        
        # Find biggest surprises (drivers predicted much better/worse than grid position)
        surprises = []
        for pred in result.predictions:
            # Assume grid position is available in metadata or use predicted position as baseline
            grid_pos = pred.predicted_position  # Simplified for demo
            surprise_factor = abs(pred.predicted_position - grid_pos)
            
            if surprise_factor > 5:  # Significant position change
                surprises.append({
                    'driver': getattr(pred, 'driver_name', pred.driver_id),
                    'predicted_position': pred.predicted_position,
                    'surprise_factor': surprise_factor
                })
        
        insights['biggest_surprises'] = sorted(surprises, key=lambda x: x['surprise_factor'], reverse=True)[:3]
        
        # Find closest battles (drivers with similar predicted positions)
        battles = []
        sorted_preds = sorted(result.predictions, key=lambda x: x.predicted_position)
        
        for i in range(len(sorted_preds) - 1):
            pos_diff = abs(sorted_preds[i+1].predicted_position - sorted_preds[i].predicted_position)
            if pos_diff < 1.5:  # Very close predictions
                battles.append({
                    'drivers': [getattr(sorted_preds[i], 'driver_name', sorted_preds[i].driver_id), 
                               getattr(sorted_preds[i+1], 'driver_name', sorted_preds[i+1].driver_id)],
                    'positions': [sorted_preds[i].predicted_position, sorted_preds[i+1].predicted_position],
                    'closeness': 1.5 - pos_diff
                })
        
        insights['closest_battles'] = sorted(battles, key=lambda x: x['closeness'], reverse=True)[:3]
        
        # Confidence distribution
        confidence_levels = {'high': 0, 'medium': 0, 'low': 0}
        for pred in result.predictions:
            confidence = pred.confidence_score
            if confidence > 0.8:
                confidence_levels['high'] += 1
            elif confidence > 0.6:
                confidence_levels['medium'] += 1
            else:
                confidence_levels['low'] += 1
        
        insights['confidence_distribution'] = confidence_levels
        
        return insights
    
    def update_model_accuracy(self, actual_results: List[int], predicted_results: List[int]):
        """
        Update model accuracy tracking with actual race results.
        
        Args:
            actual_results: Actual finishing positions
            predicted_results: Predicted finishing positions
        """
        if len(actual_results) != len(predicted_results):
            logger.warning("Actual and predicted results length mismatch")
            return
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(np.array(actual_results) - np.array(predicted_results)))
        accuracy_score = max(0.0, 1.0 - mae / 10.0)  # Normalize
        
        # Update running average of accuracy
        if self.model_type in self.model_accuracies:
            current_accuracy = self.model_accuracies[self.model_type]
            # Weighted average (give more weight to recent results)
            self.model_accuracies[self.model_type] = 0.7 * current_accuracy + 0.3 * accuracy_score
        else:
            self.model_accuracies[self.model_type] = accuracy_score
        
        logger.info(f"Updated model accuracy: {self.model_accuracies[self.model_type]:.3f}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance metrics."""
        return {
            'model_type': self.model_type,
            'model_initialized': self.model is not None,
            'model_accuracies': self.model_accuracies,
            'cache_size': len(self.prediction_cache),
            'cache_ttl': self.cache_ttl,
            'last_prediction': max([data['timestamp'] for data in self.prediction_cache.values()]) 
                              if self.prediction_cache else None
        }