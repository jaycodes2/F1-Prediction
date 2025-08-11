"""
Results formatting and insight generation for F1 race predictions.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json

from .prediction_engine import PredictionResult, PredictionRequest
from ..models.data_models import PositionPrediction


logger = logging.getLogger(__name__)


@dataclass
class OvertakingProbability:
    """Probability of one driver overtaking another."""
    overtaker_id: str
    overtaker_name: str
    target_id: str
    target_name: str
    probability: float
    expected_lap: Optional[int] = None
    factors: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RaceInsight:
    """Individual race insight with supporting data."""
    insight_type: str  # 'winner', 'surprise', 'battle', 'strategy', 'weather'
    title: str
    description: str
    confidence: float
    supporting_data: Dict[str, Any]
    drivers_involved: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FormattedRaceResult:
    """Comprehensive formatted race prediction result."""
    race_info: Dict[str, Any]
    position_predictions: List[Dict[str, Any]]
    podium_analysis: Dict[str, Any]
    overtaking_analysis: Dict[str, Any]
    key_insights: List[RaceInsight]
    statistical_summary: Dict[str, Any]
    confidence_analysis: Dict[str, Any]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['generated_at'] = self.generated_at.isoformat()
        data['key_insights'] = [insight.to_dict() for insight in self.key_insights]
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ResultsFormatter:
    """
    Advanced results formatter for F1 race predictions.
    """
    
    def __init__(self):
        self.f1_points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # Track characteristics for insight generation
        self.track_characteristics = {
            'Monaco': {'overtaking_difficulty': 0.9, 'qualifying_importance': 0.95, 'weather_impact': 0.3},
            'Silverstone': {'overtaking_difficulty': 0.3, 'qualifying_importance': 0.6, 'weather_impact': 0.8},
            'Monza': {'overtaking_difficulty': 0.2, 'qualifying_importance': 0.5, 'weather_impact': 0.4},
            'Spa': {'overtaking_difficulty': 0.4, 'qualifying_importance': 0.7, 'weather_impact': 0.9},
            'Suzuka': {'overtaking_difficulty': 0.6, 'qualifying_importance': 0.8, 'weather_impact': 0.7}
        }
    
    def format_race_result(self, prediction_result: PredictionResult, 
                          prediction_request: PredictionRequest) -> FormattedRaceResult:
        """
        Format a race prediction result into a comprehensive analysis.
        
        Args:
            prediction_result: Raw prediction result
            prediction_request: Original prediction request
            
        Returns:
            Formatted race result with insights
        """
        logger.info(f"Formatting results for {prediction_result.race_name}")
        
        # Extract race information
        race_info = self._extract_race_info(prediction_request, prediction_result)
        
        # Format position predictions
        position_predictions = self._format_position_predictions(prediction_result.predictions)
        
        # Analyze podium probabilities
        podium_analysis = self._analyze_podium_probabilities(prediction_result.predictions)
        
        # Calculate overtaking probabilities
        overtaking_analysis = self._analyze_overtaking_probabilities(
            prediction_result.predictions, prediction_request
        )
        
        # Generate key insights
        key_insights = self._generate_key_insights(
            prediction_result, prediction_request
        )
        
        # Create statistical summary
        statistical_summary = self._create_statistical_summary(prediction_result)
        
        # Analyze confidence
        confidence_analysis = self._analyze_confidence_distribution(prediction_result)
        
        formatted_result = FormattedRaceResult(
            race_info=race_info,
            position_predictions=position_predictions,
            podium_analysis=podium_analysis,
            overtaking_analysis=overtaking_analysis,
            key_insights=key_insights,
            statistical_summary=statistical_summary,
            confidence_analysis=confidence_analysis,
            generated_at=datetime.now()
        )
        
        logger.info(f"Results formatted successfully with {len(key_insights)} insights")
        return formatted_result
    
    def _extract_race_info(self, request: PredictionRequest, 
                          result: PredictionResult) -> Dict[str, Any]:
        """Extract and format race information."""
        return {
            'race_name': request.race_name,
            'circuit': request.circuit,
            'date': request.date.isoformat(),
            'session_type': request.session_type,
            'total_drivers': len(request.drivers),
            'weather_conditions': request.weather,
            'overall_confidence': result.confidence_score,
            'prediction_model': result.prediction_metadata.get('model_type', 'Unknown')
        }
    
    def _format_position_predictions(self, predictions: List[PositionPrediction]) -> List[Dict[str, Any]]:
        """Format individual position predictions."""
        formatted_predictions = []
        
        for i, pred in enumerate(predictions):
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            
            # Calculate podium probability (top 3)
            podium_prob = sum(pred.probability_distribution[:3]) if len(pred.probability_distribution) >= 3 else 0
            
            # Calculate points probability (top 10)
            points_prob = sum(pred.probability_distribution[:10]) if len(pred.probability_distribution) >= 10 else 0
            
            formatted_pred = {
                'position': i + 1,
                'driver_id': pred.driver_id,
                'driver_name': driver_name,
                'predicted_position': pred.predicted_position,
                'expected_points': pred.expected_points,
                'confidence_score': pred.confidence_score,
                'podium_probability': podium_prob,
                'points_probability': points_prob,
                'position_range': self._calculate_position_range(pred.probability_distribution),
                'key_strengths': self._identify_driver_strengths(pred),
                'risk_factors': self._identify_risk_factors(pred)
            }
            
            formatted_predictions.append(formatted_pred)
        
        return formatted_predictions
    
    def _calculate_position_range(self, prob_distribution: List[float]) -> Dict[str, int]:
        """Calculate likely position range for a driver."""
        if not prob_distribution:
            return {'min': 1, 'max': 20, 'most_likely': 10}
        
        # Find positions with significant probability (>5%)
        significant_positions = [i + 1 for i, prob in enumerate(prob_distribution) if prob > 0.05]
        
        if not significant_positions:
            # Fallback to highest probability positions
            top_indices = np.argsort(prob_distribution)[-3:]  # Top 3 probabilities
            significant_positions = [i + 1 for i in top_indices]
        
        most_likely = significant_positions[np.argmax([prob_distribution[pos-1] for pos in significant_positions])]
        
        return {
            'min': min(significant_positions),
            'max': max(significant_positions),
            'most_likely': most_likely
        }
    
    def _identify_driver_strengths(self, prediction: PositionPrediction) -> List[str]:
        """Identify key strengths contributing to the prediction."""
        strengths = []
        
        # High confidence indicates strong prediction factors
        if prediction.confidence_score > 0.8:
            strengths.append("Strong overall performance indicators")
        
        # High expected points indicates good finishing position
        if prediction.expected_points > 15:
            strengths.append("Excellent championship position potential")
        elif prediction.expected_points > 8:
            strengths.append("Good points scoring opportunity")
        
        # Predicted position analysis
        if prediction.predicted_position <= 3:
            strengths.append("Podium contender")
        elif prediction.predicted_position <= 6:
            strengths.append("Top 6 finish likely")
        elif prediction.predicted_position <= 10:
            strengths.append("Points finish expected")
        
        return strengths
    
    def _identify_risk_factors(self, prediction: PositionPrediction) -> List[str]:
        """Identify potential risk factors for the prediction."""
        risks = []
        
        # Low confidence indicates uncertainty
        if prediction.confidence_score < 0.5:
            risks.append("High prediction uncertainty")
        elif prediction.confidence_score < 0.7:
            risks.append("Moderate prediction uncertainty")
        
        # Position-based risks
        if prediction.predicted_position > 15:
            risks.append("Risk of non-finish")
        elif prediction.predicted_position > 10:
            risks.append("Outside points positions")
        
        # Expected points analysis
        if prediction.expected_points < 1:
            risks.append("Low points scoring probability")
        
        return risks
    
    def _analyze_podium_probabilities(self, predictions: List[PositionPrediction]) -> Dict[str, Any]:
        """Analyze podium finishing probabilities."""
        podium_analysis = {
            'most_likely_podium': [],
            'podium_probabilities': {},
            'podium_battles': [],
            'surprise_podium_candidates': []
        }
        
        # Calculate individual podium probabilities
        for pred in predictions:
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            
            # Podium probability (positions 1-3)
            podium_prob = sum(pred.probability_distribution[:3]) if len(pred.probability_distribution) >= 3 else 0
            podium_analysis['podium_probabilities'][driver_name] = podium_prob
        
        # Most likely podium (top 3 by podium probability)
        sorted_podium = sorted(
            podium_analysis['podium_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        podium_analysis['most_likely_podium'] = [driver for driver, prob in sorted_podium[:3]]
        
        # Identify close podium battles (drivers with similar probabilities)
        for i in range(len(sorted_podium) - 1):
            driver1, prob1 = sorted_podium[i]
            driver2, prob2 = sorted_podium[i + 1]
            
            if abs(prob1 - prob2) < 0.1 and prob1 > 0.2:  # Close battle with significant probability
                podium_analysis['podium_battles'].append({
                    'drivers': [driver1, driver2],
                    'probabilities': [prob1, prob2],
                    'closeness': abs(prob1 - prob2)
                })
        
        # Surprise podium candidates (high podium probability despite lower predicted position)
        for pred in predictions:
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            podium_prob = podium_analysis['podium_probabilities'][driver_name]
            
            if pred.predicted_position > 6 and podium_prob > 0.15:
                podium_analysis['surprise_podium_candidates'].append({
                    'driver': driver_name,
                    'predicted_position': pred.predicted_position,
                    'podium_probability': podium_prob
                })
        
        return podium_analysis
    
    def _analyze_overtaking_probabilities(self, predictions: List[PositionPrediction], 
                                        request: PredictionRequest) -> Dict[str, Any]:
        """Analyze overtaking probabilities and scenarios."""
        overtaking_analysis = {
            'most_likely_overtakes': [],
            'overtaking_hotspots': [],
            'defensive_battles': [],
            'track_overtaking_factor': 0.5
        }
        
        # Get track characteristics
        circuit_name = request.circuit.split()[-1] if request.circuit else 'Unknown'
        track_chars = self.track_characteristics.get(circuit_name, {
            'overtaking_difficulty': 0.5, 'qualifying_importance': 0.7, 'weather_impact': 0.5
        })
        
        overtaking_analysis['track_overtaking_factor'] = 1.0 - track_chars['overtaking_difficulty']
        
        # Analyze potential overtakes based on grid vs predicted positions
        overtaking_probabilities = []
        
        for i, pred in enumerate(predictions):
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            
            # Find original grid position
            grid_position = None
            for driver in request.drivers:
                if driver.get('driver_id') == pred.driver_id or driver.get('name') == driver_name:
                    grid_position = driver.get('grid_position', i + 1)
                    break
            
            if grid_position is None:
                grid_position = i + 1
            
            # Calculate overtaking potential
            position_gain = grid_position - pred.predicted_position
            
            if position_gain > 2:  # Significant position gain
                # Find drivers being overtaken
                for j, target_pred in enumerate(predictions):
                    target_name = getattr(target_pred, 'driver_name', target_pred.driver_id)
                    target_grid = None
                    
                    for driver in request.drivers:
                        if (driver.get('driver_id') == target_pred.driver_id or 
                            driver.get('name') == target_name):
                            target_grid = driver.get('grid_position', j + 1)
                            break
                    
                    if target_grid is None:
                        target_grid = j + 1
                    
                    # Check if overtaking is likely
                    if (target_grid < grid_position and 
                        target_pred.predicted_position > pred.predicted_position):
                        
                        overtake_prob = min(0.9, position_gain * 0.2 * overtaking_analysis['track_overtaking_factor'])
                        
                        overtaking_probabilities.append(OvertakingProbability(
                            overtaker_id=pred.driver_id,
                            overtaker_name=driver_name,
                            target_id=target_pred.driver_id,
                            target_name=target_name,
                            probability=overtake_prob,
                            factors=[
                                f"Grid position advantage: {grid_position} -> {pred.predicted_position}",
                                f"Track overtaking factor: {overtaking_analysis['track_overtaking_factor']:.2f}"
                            ]
                        ))
        
        # Sort by probability and take top overtakes
        overtaking_probabilities.sort(key=lambda x: x.probability, reverse=True)
        overtaking_analysis['most_likely_overtakes'] = [
            overtake.to_dict() for overtake in overtaking_probabilities[:5]
        ]
        
        # Identify overtaking hotspots (positions with multiple potential overtakes)
        position_overtakes = {}
        for overtake in overtaking_probabilities:
            pos = overtake.target_name  # Simplified - could use position ranges
            if pos not in position_overtakes:
                position_overtakes[pos] = []
            position_overtakes[pos].append(overtake)
        
        hotspots = [(pos, overtakes) for pos, overtakes in position_overtakes.items() 
                   if len(overtakes) > 1]
        overtaking_analysis['overtaking_hotspots'] = [
            {
                'position_area': pos,
                'overtake_count': len(overtakes),
                'total_probability': sum(o.probability for o in overtakes)
            }
            for pos, overtakes in hotspots
        ]
        
        return overtaking_analysis
    
    def _generate_key_insights(self, result: PredictionResult, 
                             request: PredictionRequest) -> List[RaceInsight]:
        """Generate key insights from the prediction."""
        insights = []
        
        # Winner insight
        winner = result.predictions[0]
        winner_name = getattr(winner, 'driver_name', winner.driver_id)
        
        insights.append(RaceInsight(
            insight_type='winner',
            title=f"{winner_name} Predicted to Win",
            description=f"{winner_name} has the highest probability of winning with a confidence score of {winner.confidence_score:.3f}. Expected to finish in position {winner.predicted_position} with {winner.expected_points:.1f} championship points.",
            confidence=winner.confidence_score,
            supporting_data={
                'predicted_position': winner.predicted_position,
                'expected_points': winner.expected_points,
                'win_probability': winner.probability_distribution[0] if winner.probability_distribution else 0
            },
            drivers_involved=[winner.driver_id]
        ))
        
        # Surprise performance insight
        for pred in result.predictions:
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            
            # Find grid position
            grid_position = None
            for driver in request.drivers:
                if driver.get('driver_id') == pred.driver_id or driver.get('name') == driver_name:
                    grid_position = driver.get('grid_position', pred.predicted_position)
                    break
            
            if grid_position and abs(grid_position - pred.predicted_position) > 5:
                surprise_type = "positive" if grid_position > pred.predicted_position else "negative"
                
                insights.append(RaceInsight(
                    insight_type='surprise',
                    title=f"Surprise Performance: {driver_name}",
                    description=f"{driver_name} is predicted to {'outperform' if surprise_type == 'positive' else 'underperform'} their grid position significantly, moving from P{grid_position} to P{pred.predicted_position}.",
                    confidence=pred.confidence_score,
                    supporting_data={
                        'grid_position': grid_position,
                        'predicted_position': pred.predicted_position,
                        'position_change': grid_position - pred.predicted_position,
                        'surprise_type': surprise_type
                    },
                    drivers_involved=[pred.driver_id]
                ))
        
        # Weather impact insight
        weather = request.weather
        if weather.get('conditions') != 'dry':
            insights.append(RaceInsight(
                insight_type='weather',
                title="Weather Impact on Race",
                description=f"Wet conditions expected with {weather.get('humidity', 0)}% humidity. This could significantly impact race dynamics and create opportunities for skilled wet-weather drivers.",
                confidence=0.8,
                supporting_data={
                    'conditions': weather.get('conditions'),
                    'humidity': weather.get('humidity'),
                    'track_temp': weather.get('track_temp'),
                    'grip_level': weather.get('grip_level', 1.0)
                },
                drivers_involved=[]
            ))
        
        # Close battle insight
        close_battles = []
        sorted_preds = sorted(result.predictions, key=lambda x: x.predicted_position)
        
        for i in range(len(sorted_preds) - 1):
            pred1, pred2 = sorted_preds[i], sorted_preds[i + 1]
            if abs(pred1.predicted_position - pred2.predicted_position) < 1.5:
                name1 = getattr(pred1, 'driver_name', pred1.driver_id)
                name2 = getattr(pred2, 'driver_name', pred2.driver_id)
                close_battles.append((name1, name2, pred1, pred2))
        
        if close_battles:
            battle = close_battles[0]  # Take the closest battle
            insights.append(RaceInsight(
                insight_type='battle',
                title=f"Close Battle: {battle[0]} vs {battle[1]}",
                description=f"Extremely close battle predicted between {battle[0]} (P{battle[2].predicted_position:.1f}) and {battle[1]} (P{battle[3].predicted_position:.1f}). This could be decided by strategy or race incidents.",
                confidence=min(battle[2].confidence_score, battle[3].confidence_score),
                supporting_data={
                    'driver1_position': battle[2].predicted_position,
                    'driver2_position': battle[3].predicted_position,
                    'position_difference': abs(battle[2].predicted_position - battle[3].predicted_position),
                    'combined_confidence': (battle[2].confidence_score + battle[3].confidence_score) / 2
                },
                drivers_involved=[battle[2].driver_id, battle[3].driver_id]
            ))
        
        return insights
    
    def _create_statistical_summary(self, result: PredictionResult) -> Dict[str, Any]:
        """Create statistical summary of predictions."""
        predictions = result.predictions
        
        # Position statistics
        positions = [pred.predicted_position for pred in predictions]
        expected_points = [pred.expected_points for pred in predictions]
        confidences = [pred.confidence_score for pred in predictions]
        
        return {
            'total_drivers': len(predictions),
            'position_statistics': {
                'mean_position': np.mean(positions),
                'position_spread': np.std(positions),
                'median_position': np.median(positions)
            },
            'points_statistics': {
                'total_expected_points': sum(expected_points),
                'mean_expected_points': np.mean(expected_points),
                'points_scoring_drivers': sum(1 for points in expected_points if points > 0)
            },
            'confidence_statistics': {
                'mean_confidence': np.mean(confidences),
                'confidence_spread': np.std(confidences),
                'high_confidence_predictions': sum(1 for conf in confidences if conf > 0.8),
                'low_confidence_predictions': sum(1 for conf in confidences if conf < 0.5)
            },
            'prediction_quality': {
                'overall_confidence': result.confidence_score,
                'prediction_certainty': 'high' if result.confidence_score > 0.8 else 'medium' if result.confidence_score > 0.6 else 'low'
            }
        }
    
    def _analyze_confidence_distribution(self, result: PredictionResult) -> Dict[str, Any]:
        """Analyze confidence distribution across predictions."""
        confidences = [pred.confidence_score for pred in result.predictions]
        
        # Confidence buckets
        high_conf = [c for c in confidences if c > 0.8]
        medium_conf = [c for c in confidences if 0.5 <= c <= 0.8]
        low_conf = [c for c in confidences if c < 0.5]
        
        return {
            'distribution': {
                'high_confidence': {
                    'count': len(high_conf),
                    'percentage': len(high_conf) / len(confidences) * 100,
                    'average': np.mean(high_conf) if high_conf else 0
                },
                'medium_confidence': {
                    'count': len(medium_conf),
                    'percentage': len(medium_conf) / len(confidences) * 100,
                    'average': np.mean(medium_conf) if medium_conf else 0
                },
                'low_confidence': {
                    'count': len(low_conf),
                    'percentage': len(low_conf) / len(confidences) * 100,
                    'average': np.mean(low_conf) if low_conf else 0
                }
            },
            'overall_metrics': {
                'mean_confidence': np.mean(confidences),
                'confidence_variance': np.var(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences)
            },
            'reliability_indicators': {
                'consistent_predictions': len(high_conf) > len(confidences) * 0.6,
                'uncertain_predictions': len(low_conf) > len(confidences) * 0.3,
                'prediction_reliability': 'high' if np.mean(confidences) > 0.75 else 'medium' if np.mean(confidences) > 0.5 else 'low'
            }
        }


class InsightGenerator:
    """
    Advanced insight generator for F1 race predictions.
    """
    
    def __init__(self):
        self.insight_templates = {
            'dominant_performance': "Based on current form and car performance, {driver} is showing dominant pace and is highly likely to control the race from the front.",
            'comeback_drive': "{driver} starting from P{grid_pos} has excellent overtaking potential and could finish as high as P{predicted_pos}.",
            'strategy_battle': "The battle between {driver1} and {driver2} will likely be decided by pit strategy, with both drivers having similar pace.",
            'weather_wildcard': "Changing weather conditions could significantly impact the race outcome, particularly benefiting drivers skilled in wet conditions.",
            'rookie_performance': "{driver} is predicted to deliver an impressive performance, potentially outperforming more experienced drivers.",
            'home_advantage': "{driver} racing at their home circuit could benefit from local knowledge and crowd support.",
            'championship_implications': "This race could be crucial for the championship battle, with {driver} needing a strong result to maintain their position."
        }
    
    def generate_advanced_insights(self, formatted_result: FormattedRaceResult, 
                                 historical_data: Optional[Dict] = None) -> List[RaceInsight]:
        """Generate advanced insights using historical data and race context."""
        advanced_insights = []
        
        # Analyze championship implications
        championship_insight = self._analyze_championship_implications(formatted_result)
        if championship_insight:
            advanced_insights.append(championship_insight)
        
        # Analyze team battles
        team_battle_insights = self._analyze_team_battles(formatted_result)
        advanced_insights.extend(team_battle_insights)
        
        # Analyze rookie vs veteran dynamics
        experience_insights = self._analyze_experience_dynamics(formatted_result)
        advanced_insights.extend(experience_insights)
        
        # Analyze strategic opportunities
        strategy_insights = self._analyze_strategic_opportunities(formatted_result)
        advanced_insights.extend(strategy_insights)
        
        return advanced_insights
    
    def _analyze_championship_implications(self, result: FormattedRaceResult) -> Optional[RaceInsight]:
        """Analyze championship implications of the race."""
        # This would require championship standings data
        # For now, return a generic insight for top performers
        
        top_3 = result.position_predictions[:3]
        if top_3[0]['expected_points'] > 20:  # Likely winner with high points
            return RaceInsight(
                insight_type='championship',
                title="Championship Impact",
                description=f"This race could be crucial for the championship battle, with {top_3[0]['driver_name']} positioned to gain significant points.",
                confidence=0.8,
                supporting_data={
                    'expected_points': top_3[0]['expected_points'],
                    'position': top_3[0]['predicted_position']
                },
                drivers_involved=[top_3[0]['driver_id']]
            )
        
        return None
    
    def _analyze_team_battles(self, result: FormattedRaceResult) -> List[RaceInsight]:
        """Analyze battles between teammates."""
        insights = []
        
        # Group drivers by team (simplified - would need team data)
        # For demo, look for drivers with similar names or IDs
        team_battles = []
        
        predictions = result.position_predictions
        for i in range(len(predictions) - 1):
            for j in range(i + 1, len(predictions)):
                driver1 = predictions[i]
                driver2 = predictions[j]
                
                # Simple heuristic: similar team if position difference is small and both have good confidence
                if (abs(driver1['predicted_position'] - driver2['predicted_position']) < 3 and
                    driver1['confidence_score'] > 0.7 and driver2['confidence_score'] > 0.7):
                    
                    team_battles.append((driver1, driver2))
        
        for battle in team_battles[:2]:  # Limit to top 2 battles
            insights.append(RaceInsight(
                insight_type='team_battle',
                title=f"Intra-Team Battle: {battle[0]['driver_name']} vs {battle[1]['driver_name']}",
                description=f"Close battle expected between teammates, with {battle[0]['driver_name']} predicted P{battle[0]['predicted_position']} and {battle[1]['driver_name']} predicted P{battle[1]['predicted_position']}.",
                confidence=min(battle[0]['confidence_score'], battle[1]['confidence_score']),
                supporting_data={
                    'position_gap': abs(battle[0]['predicted_position'] - battle[1]['predicted_position']),
                    'points_difference': abs(battle[0]['expected_points'] - battle[1]['expected_points'])
                },
                drivers_involved=[battle[0]['driver_id'], battle[1]['driver_id']]
            ))
        
        return insights
    
    def _analyze_experience_dynamics(self, result: FormattedRaceResult) -> List[RaceInsight]:
        """Analyze rookie vs veteran performance dynamics."""
        insights = []
        
        # This would require driver experience data
        # For demo, assume drivers with lower IDs are more experienced
        predictions = result.position_predictions
        
        # Look for potential rookie outperformances
        for pred in predictions:
            if (pred['predicted_position'] <= 8 and  # Good finishing position
                pred['confidence_score'] > 0.7 and  # High confidence
                'rookie' in pred['driver_name'].lower()):  # Simple heuristic
                
                insights.append(RaceInsight(
                    insight_type='rookie_performance',
                    title=f"Impressive Rookie Performance: {pred['driver_name']}",
                    description=f"{pred['driver_name']} is predicted to deliver an outstanding performance, finishing P{pred['predicted_position']} and potentially outperforming more experienced drivers.",
                    confidence=pred['confidence_score'],
                    supporting_data={
                        'predicted_position': pred['predicted_position'],
                        'expected_points': pred['expected_points']
                    },
                    drivers_involved=[pred['driver_id']]
                ))
        
        return insights
    
    def _analyze_strategic_opportunities(self, result: FormattedRaceResult) -> List[RaceInsight]:
        """Analyze strategic opportunities in the race."""
        insights = []
        
        # Look for drivers with high overtaking potential
        overtaking_analysis = result.overtaking_analysis
        
        if overtaking_analysis['most_likely_overtakes']:
            top_overtake = overtaking_analysis['most_likely_overtakes'][0]
            
            insights.append(RaceInsight(
                insight_type='strategy',
                title="Key Strategic Opportunity",
                description=f"Strategic pit timing could be crucial, with {top_overtake['overtaker_name']} having a {top_overtake['probability']:.1%} chance of overtaking {top_overtake['target_name']}.",
                confidence=top_overtake['probability'],
                supporting_data=top_overtake,
                drivers_involved=[top_overtake['overtaker_id'], top_overtake['target_id']]
            ))
        
        return insights