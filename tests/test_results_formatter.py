"""
Tests for results formatting and insight generation.
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.services.results_formatter import (
    ResultsFormatter, InsightGenerator, OvertakingProbability, 
    RaceInsight, FormattedRaceResult
)
from src.services.prediction_engine import PredictionResult, PredictionRequest
from src.models.data_models import PositionPrediction


class TestOvertakingProbability:
    """Tests for OvertakingProbability class."""
    
    def test_overtaking_probability_creation(self):
        """Test overtaking probability creation."""
        overtake = OvertakingProbability(
            overtaker_id='VER',
            overtaker_name='Max Verstappen',
            target_id='HAM',
            target_name='Lewis Hamilton',
            probability=0.75,
            expected_lap=25,
            factors=['Superior pace', 'DRS advantage']
        )
        
        assert overtake.overtaker_id == 'VER'
        assert overtake.overtaker_name == 'Max Verstappen'
        assert overtake.probability == 0.75
        assert overtake.expected_lap == 25
        assert len(overtake.factors) == 2
    
    def test_overtaking_probability_to_dict(self):
        """Test overtaking probability serialization."""
        overtake = OvertakingProbability(
            overtaker_id='VER',
            overtaker_name='Max Verstappen',
            target_id='HAM',
            target_name='Lewis Hamilton',
            probability=0.75
        )
        
        data = overtake.to_dict()
        
        assert isinstance(data, dict)
        assert data['overtaker_id'] == 'VER'
        assert data['probability'] == 0.75


class TestRaceInsight:
    """Tests for RaceInsight class."""
    
    def test_race_insight_creation(self):
        """Test race insight creation."""
        insight = RaceInsight(
            insight_type='winner',
            title='Verstappen Predicted to Win',
            description='Max Verstappen has the highest probability of winning.',
            confidence=0.85,
            supporting_data={'win_probability': 0.45},
            drivers_involved=['VER']
        )
        
        assert insight.insight_type == 'winner'
        assert insight.title == 'Verstappen Predicted to Win'
        assert insight.confidence == 0.85
        assert len(insight.drivers_involved) == 1
    
    def test_race_insight_to_dict(self):
        """Test race insight serialization."""
        insight = RaceInsight(
            insight_type='battle',
            title='Close Battle',
            description='Very close battle expected.',
            confidence=0.7,
            supporting_data={'gap': 0.1},
            drivers_involved=['VER', 'HAM']
        )
        
        data = insight.to_dict()
        
        assert isinstance(data, dict)
        assert data['insight_type'] == 'battle'
        assert data['confidence'] == 0.7
        assert len(data['drivers_involved']) == 2


class TestResultsFormatter:
    """Tests for ResultsFormatter class."""
    
    @pytest.fixture
    def formatter(self):
        """Create test results formatter."""
        return ResultsFormatter()
    
    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result."""
        predictions = []
        
        for i in range(5):
            # Create probability distribution
            prob_dist = [0.0] * 20
            prob_dist[i] = 0.6  # High probability for predicted position
            prob_dist[i+1] = 0.3 if i < 19 else 0.0
            prob_dist[i-1] = 0.1 if i > 0 else 0.0
            
            pred = PositionPrediction(
                driver_id=f'D{i+1}',
                predicted_position=i + 1,
                probability_distribution=prob_dist,
                expected_points=max(0, 26 - (i+1) * 2),  # Decreasing points
                confidence_score=0.8 - i * 0.1
            )
            # Add driver name as attribute
            pred.driver_name = f'Driver {i+1}'
            predictions.append(pred)
        
        return PredictionResult(
            race_name='Test Grand Prix',
            predictions=predictions,
            confidence_score=0.75,
            prediction_metadata={'model_type': 'test_model'},
            generated_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_prediction_request(self):
        """Create sample prediction request."""
        return PredictionRequest(
            race_name='Test Grand Prix',
            circuit='Test Circuit',
            date=datetime.now(),
            drivers=[
                {
                    'driver_id': f'D{i+1}',
                    'name': f'Driver {i+1}',
                    'grid_position': i + 1
                }
                for i in range(5)
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 30.0,
                'air_temp': 25.0,
                'humidity': 60.0
            }
        )
    
    def test_formatter_initialization(self, formatter):
        """Test formatter initialization."""
        assert formatter.f1_points_system is not None
        assert len(formatter.f1_points_system) == 10
        assert formatter.f1_points_system[1] == 25
        assert formatter.track_characteristics is not None
    
    def test_format_race_result(self, formatter, sample_prediction_result, sample_prediction_request):
        """Test complete race result formatting."""
        formatted_result = formatter.format_race_result(
            sample_prediction_result, sample_prediction_request
        )
        
        assert isinstance(formatted_result, FormattedRaceResult)
        assert formatted_result.race_info['race_name'] == 'Test Grand Prix'
        assert len(formatted_result.position_predictions) == 5
        assert formatted_result.podium_analysis is not None
        assert formatted_result.overtaking_analysis is not None
        assert len(formatted_result.key_insights) > 0
        assert formatted_result.statistical_summary is not None
        assert formatted_result.confidence_analysis is not None
    
    def test_extract_race_info(self, formatter, sample_prediction_request, sample_prediction_result):
        """Test race information extraction."""
        race_info = formatter._extract_race_info(sample_prediction_request, sample_prediction_result)
        
        assert race_info['race_name'] == 'Test Grand Prix'
        assert race_info['circuit'] == 'Test Circuit'
        assert race_info['total_drivers'] == 5
        assert race_info['overall_confidence'] == 0.75
        assert 'weather_conditions' in race_info
    
    def test_format_position_predictions(self, formatter, sample_prediction_result):
        """Test position predictions formatting."""
        formatted_preds = formatter._format_position_predictions(sample_prediction_result.predictions)
        
        assert len(formatted_preds) == 5
        
        for i, pred in enumerate(formatted_preds):
            assert pred['position'] == i + 1
            assert pred['driver_id'] == f'D{i+1}'
            assert pred['driver_name'] == f'Driver {i+1}'
            assert 'podium_probability' in pred
            assert 'points_probability' in pred
            assert 'position_range' in pred
            assert 'key_strengths' in pred
            assert 'risk_factors' in pred
    
    def test_calculate_position_range(self, formatter):
        """Test position range calculation."""
        # High probability for positions 3-5
        prob_dist = [0.0] * 20
        prob_dist[2] = 0.4  # Position 3
        prob_dist[3] = 0.3  # Position 4
        prob_dist[4] = 0.2  # Position 5
        prob_dist[5] = 0.1  # Position 6
        
        position_range = formatter._calculate_position_range(prob_dist)
        
        assert 'min' in position_range
        assert 'max' in position_range
        assert 'most_likely' in position_range
        assert position_range['min'] <= position_range['max']
        assert position_range['most_likely'] == 3  # Highest probability
    
    def test_identify_driver_strengths(self, formatter):
        """Test driver strengths identification."""
        # High-performing prediction
        pred = PositionPrediction(
            driver_id='VER',
            predicted_position=2,
            probability_distribution=[0.0] * 20,
            expected_points=18.0,
            confidence_score=0.9
        )
        
        strengths = formatter._identify_driver_strengths(pred)
        
        assert isinstance(strengths, list)
        assert len(strengths) > 0
        assert any('Strong overall performance' in strength for strength in strengths)
        assert any('championship position' in strength for strength in strengths)
    
    def test_identify_risk_factors(self, formatter):
        """Test risk factors identification."""
        # Low-performing prediction
        pred = PositionPrediction(
            driver_id='BOT',
            predicted_position=16,
            probability_distribution=[0.0] * 20,
            expected_points=0.0,
            confidence_score=0.3
        )
        
        risks = formatter._identify_risk_factors(pred)
        
        assert isinstance(risks, list)
        assert len(risks) > 0
        assert any('uncertainty' in risk for risk in risks)
        assert any('points' in risk for risk in risks)
    
    def test_analyze_podium_probabilities(self, formatter, sample_prediction_result):
        """Test podium probability analysis."""
        podium_analysis = formatter._analyze_podium_probabilities(sample_prediction_result.predictions)
        
        assert 'most_likely_podium' in podium_analysis
        assert 'podium_probabilities' in podium_analysis
        assert 'podium_battles' in podium_analysis
        assert 'surprise_podium_candidates' in podium_analysis
        
        assert len(podium_analysis['most_likely_podium']) == 3
        assert len(podium_analysis['podium_probabilities']) == 5
    
    def test_analyze_overtaking_probabilities(self, formatter, sample_prediction_result, sample_prediction_request):
        """Test overtaking probability analysis."""
        overtaking_analysis = formatter._analyze_overtaking_probabilities(
            sample_prediction_result.predictions, sample_prediction_request
        )
        
        assert 'most_likely_overtakes' in overtaking_analysis
        assert 'overtaking_hotspots' in overtaking_analysis
        assert 'defensive_battles' in overtaking_analysis
        assert 'track_overtaking_factor' in overtaking_analysis
        
        assert isinstance(overtaking_analysis['most_likely_overtakes'], list)
        assert 0 <= overtaking_analysis['track_overtaking_factor'] <= 1
    
    def test_generate_key_insights(self, formatter, sample_prediction_result, sample_prediction_request):
        """Test key insights generation."""
        insights = formatter._generate_key_insights(sample_prediction_result, sample_prediction_request)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Should have at least a winner insight
        winner_insights = [i for i in insights if i.insight_type == 'winner']
        assert len(winner_insights) > 0
        
        # Check insight structure
        for insight in insights:
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'confidence')
            assert hasattr(insight, 'supporting_data')
            assert hasattr(insight, 'drivers_involved')
    
    def test_create_statistical_summary(self, formatter, sample_prediction_result):
        """Test statistical summary creation."""
        summary = formatter._create_statistical_summary(sample_prediction_result)
        
        assert 'total_drivers' in summary
        assert 'position_statistics' in summary
        assert 'points_statistics' in summary
        assert 'confidence_statistics' in summary
        assert 'prediction_quality' in summary
        
        assert summary['total_drivers'] == 5
        
        # Check position statistics
        pos_stats = summary['position_statistics']
        assert 'mean_position' in pos_stats
        assert 'position_spread' in pos_stats
        assert 'median_position' in pos_stats
        
        # Check points statistics
        points_stats = summary['points_statistics']
        assert 'total_expected_points' in points_stats
        assert 'points_scoring_drivers' in points_stats
    
    def test_analyze_confidence_distribution(self, formatter, sample_prediction_result):
        """Test confidence distribution analysis."""
        confidence_analysis = formatter._analyze_confidence_distribution(sample_prediction_result)
        
        assert 'distribution' in confidence_analysis
        assert 'overall_metrics' in confidence_analysis
        assert 'reliability_indicators' in confidence_analysis
        
        # Check distribution
        distribution = confidence_analysis['distribution']
        assert 'high_confidence' in distribution
        assert 'medium_confidence' in distribution
        assert 'low_confidence' in distribution
        
        # Check that percentages add up to 100
        total_percentage = (
            distribution['high_confidence']['percentage'] +
            distribution['medium_confidence']['percentage'] +
            distribution['low_confidence']['percentage']
        )
        assert abs(total_percentage - 100.0) < 0.01
    
    def test_formatted_result_serialization(self, formatter, sample_prediction_result, sample_prediction_request):
        """Test formatted result serialization."""
        formatted_result = formatter.format_race_result(
            sample_prediction_result, sample_prediction_request
        )
        
        # Test to_dict
        result_dict = formatted_result.to_dict()
        assert isinstance(result_dict, dict)
        assert 'race_info' in result_dict
        assert 'position_predictions' in result_dict
        assert 'key_insights' in result_dict
        
        # Test to_json
        result_json = formatted_result.to_json()
        assert isinstance(result_json, str)
        assert 'race_info' in result_json
        
        # Should be valid JSON
        import json
        parsed = json.loads(result_json)
        assert isinstance(parsed, dict)


class TestInsightGenerator:
    """Tests for InsightGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create test insight generator."""
        return InsightGenerator()
    
    @pytest.fixture
    def sample_formatted_result(self):
        """Create sample formatted result."""
        # Create a mock formatted result
        formatted_result = Mock(spec=FormattedRaceResult)
        
        # Mock position predictions
        formatted_result.position_predictions = [
            {
                'driver_id': 'VER',
                'driver_name': 'Max Verstappen',
                'predicted_position': 1,
                'expected_points': 25.0,
                'confidence_score': 0.9
            },
            {
                'driver_id': 'HAM',
                'driver_name': 'Lewis Hamilton',
                'predicted_position': 2,
                'expected_points': 18.0,
                'confidence_score': 0.85
            },
            {
                'driver_id': 'LEC',
                'driver_name': 'Charles Leclerc',
                'predicted_position': 3,
                'expected_points': 15.0,
                'confidence_score': 0.8
            }
        ]
        
        # Mock overtaking analysis
        formatted_result.overtaking_analysis = {
            'most_likely_overtakes': [
                {
                    'overtaker_id': 'HAM',
                    'overtaker_name': 'Lewis Hamilton',
                    'target_id': 'VER',
                    'target_name': 'Max Verstappen',
                    'probability': 0.3
                }
            ]
        }
        
        return formatted_result
    
    def test_generator_initialization(self, generator):
        """Test insight generator initialization."""
        assert generator.insight_templates is not None
        assert len(generator.insight_templates) > 0
        assert 'dominant_performance' in generator.insight_templates
    
    def test_generate_advanced_insights(self, generator, sample_formatted_result):
        """Test advanced insights generation."""
        insights = generator.generate_advanced_insights(sample_formatted_result)
        
        assert isinstance(insights, list)
        # Should generate at least some insights
        assert len(insights) >= 0
        
        # Check insight structure if any generated
        for insight in insights:
            assert isinstance(insight, RaceInsight)
            assert hasattr(insight, 'insight_type')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'description')
    
    def test_analyze_championship_implications(self, generator, sample_formatted_result):
        """Test championship implications analysis."""
        insight = generator._analyze_championship_implications(sample_formatted_result)
        
        # Should generate insight for high-scoring driver
        if insight:
            assert isinstance(insight, RaceInsight)
            assert insight.insight_type == 'championship'
            assert 'championship' in insight.title.lower()
    
    def test_analyze_team_battles(self, generator, sample_formatted_result):
        """Test team battles analysis."""
        insights = generator._analyze_team_battles(sample_formatted_result)
        
        assert isinstance(insights, list)
        
        # Check insight structure if any generated
        for insight in insights:
            assert isinstance(insight, RaceInsight)
            assert insight.insight_type == 'team_battle'
            assert len(insight.drivers_involved) == 2
    
    def test_analyze_strategic_opportunities(self, generator, sample_formatted_result):
        """Test strategic opportunities analysis."""
        insights = generator._analyze_strategic_opportunities(sample_formatted_result)
        
        assert isinstance(insights, list)
        
        # Should generate strategy insight based on overtaking data
        if insights:
            strategy_insights = [i for i in insights if i.insight_type == 'strategy']
            assert len(strategy_insights) > 0
            
            for insight in strategy_insights:
                assert 'strategic' in insight.title.lower() or 'strategy' in insight.description.lower()


class TestIntegrationResultsFormatting:
    """Integration tests for results formatting pipeline."""
    
    def test_complete_formatting_pipeline(self):
        """Test complete formatting pipeline with realistic data."""
        # Create realistic prediction result
        predictions = []
        driver_names = ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc', 'Lando Norris', 'Carlos Sainz']
        
        for i, name in enumerate(driver_names):
            # Create realistic probability distribution
            prob_dist = [0.0] * 20
            predicted_pos = i + 1
            
            # Main probability at predicted position
            prob_dist[predicted_pos - 1] = 0.4
            
            # Spread probability to nearby positions
            if predicted_pos > 1:
                prob_dist[predicted_pos - 2] = 0.2
            if predicted_pos < 20:
                prob_dist[predicted_pos] = 0.3
            if predicted_pos < 19:
                prob_dist[predicted_pos + 1] = 0.1
            
            pred = PositionPrediction(
                driver_id=f'D{i+1}',
                predicted_position=predicted_pos,
                probability_distribution=prob_dist,
                expected_points=max(0, 26 - predicted_pos * 2),
                confidence_score=0.9 - i * 0.1
            )
            pred.driver_name = name
            predictions.append(pred)
        
        prediction_result = PredictionResult(
            race_name='Monaco Grand Prix 2024',
            predictions=predictions,
            confidence_score=0.82,
            prediction_metadata={'model_type': 'ensemble'},
            generated_at=datetime.now()
        )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            race_name='Monaco Grand Prix 2024',
            circuit='Circuit de Monaco',
            date=datetime(2024, 5, 26),
            drivers=[
                {
                    'driver_id': f'D{i+1}',
                    'name': name,
                    'grid_position': i + 2  # Different from predicted to create overtaking scenarios
                }
                for i, name in enumerate(driver_names)
            ],
            weather={
                'conditions': 'dry',
                'track_temp': 35.0,
                'air_temp': 28.0,
                'humidity': 65.0
            }
        )
        
        # Format results
        formatter = ResultsFormatter()
        formatted_result = formatter.format_race_result(prediction_result, prediction_request)
        
        # Verify comprehensive formatting
        assert isinstance(formatted_result, FormattedRaceResult)
        assert formatted_result.race_info['race_name'] == 'Monaco Grand Prix 2024'
        assert len(formatted_result.position_predictions) == 5
        
        # Check podium analysis
        podium = formatted_result.podium_analysis
        assert len(podium['most_likely_podium']) == 3
        assert 'Max Verstappen' in podium['most_likely_podium']
        
        # Check overtaking analysis
        overtaking = formatted_result.overtaking_analysis
        assert 'most_likely_overtakes' in overtaking
        assert 'track_overtaking_factor' in overtaking
        
        # Check insights
        insights = formatted_result.key_insights
        assert len(insights) > 0
        
        winner_insights = [i for i in insights if i.insight_type == 'winner']
        assert len(winner_insights) > 0
        assert 'Max Verstappen' in winner_insights[0].title
        
        # Check statistical summary
        stats = formatted_result.statistical_summary
        assert stats['total_drivers'] == 5
        assert stats['points_statistics']['total_expected_points'] > 0
        
        # Check confidence analysis
        conf_analysis = formatted_result.confidence_analysis
        assert 'distribution' in conf_analysis
        assert 'reliability_indicators' in conf_analysis
        
        # Test serialization
        result_dict = formatted_result.to_dict()
        assert isinstance(result_dict, dict)
        
        result_json = formatted_result.to_json()
        assert isinstance(result_json, str)
        
        # Generate advanced insights
        generator = InsightGenerator()
        advanced_insights = generator.generate_advanced_insights(formatted_result)
        assert isinstance(advanced_insights, list)
        
        print(f"✓ Complete formatting pipeline test passed")
        print(f"  - Generated {len(insights)} key insights")
        print(f"  - Generated {len(advanced_insights)} advanced insights")
        print(f"  - Analyzed {len(overtaking['most_likely_overtakes'])} overtaking scenarios")
        print(f"  - Overall confidence: {formatted_result.race_info['overall_confidence']:.3f}")
    
    def test_edge_cases_handling(self):
        """Test handling of edge cases in formatting."""
        # Test with minimal data
        minimal_pred = PositionPrediction(
            driver_id='TEST',
            predicted_position=10,
            probability_distribution=[],  # Empty distribution
            expected_points=1.0,
            confidence_score=0.5
        )
        minimal_pred.driver_name = 'Test Driver'
        
        prediction_result = PredictionResult(
            race_name='Test Race',
            predictions=[minimal_pred],
            confidence_score=0.5,
            prediction_metadata={},
            generated_at=datetime.now()
        )
        
        prediction_request = PredictionRequest(
            race_name='Test Race',
            circuit='Test Circuit',
            date=datetime.now(),
            drivers=[{'driver_id': 'TEST', 'name': 'Test Driver', 'grid_position': 10}],
            weather={'conditions': 'unknown'}
        )
        
        # Should handle gracefully without errors
        formatter = ResultsFormatter()
        formatted_result = formatter.format_race_result(prediction_result, prediction_request)
        
        assert isinstance(formatted_result, FormattedRaceResult)
        assert len(formatted_result.position_predictions) == 1
        assert formatted_result.statistical_summary['total_drivers'] == 1