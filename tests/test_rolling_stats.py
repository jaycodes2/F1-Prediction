"""
Tests for rolling statistics calculator.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.features.rolling_stats import RollingStatsCalculator, RollingStatsError
from src.models.data_models import (
    RaceData, RaceResult, QualifyingResult, 
    WeatherData, LapTime, RaceStatus
)


class TestRollingStatsCalculator:
    """Tests for the RollingStatsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a test rolling stats calculator."""
        return RollingStatsCalculator(default_window=3)
    
    @pytest.fixture
    def sample_race_series(self):
        """Create a series of races for testing rolling statistics."""
        races = []
        base_date = datetime(2024, 3, 1)
        
        # Create 6 races with varying performance for Hamilton and Verstappen
        for i in range(6):
            weather = WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
            
            # Hamilton's performance varies over time
            hamilton_position = min(1 + i, 10)  # Gets worse over time
            hamilton_points = max(25 - i * 3, 0)
            
            # Verstappen's performance improves over time
            verstappen_position = max(10 - i, 1)  # Gets better over time
            verstappen_points = min(5 + i * 4, 25)
            
            results = [
                RaceResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    grid_position=hamilton_position,
                    final_position=hamilton_position,
                    points=hamilton_points,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                ),
                RaceResult(
                    driver_id='verstappen',
                    constructor_id='red_bull',
                    grid_position=verstappen_position,
                    final_position=verstappen_position,
                    points=verstappen_points,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                ),
                RaceResult(
                    driver_id='leclerc',
                    constructor_id='ferrari',
                    grid_position=5,
                    final_position=5,
                    points=10,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                )
            ]
            
            qualifying = [
                QualifyingResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    position=hamilton_position,
                    q1_time=LapTime(91000, 1),
                    q2_time=LapTime(90500, 1),
                    q3_time=LapTime(90000, 1)
                ),
                QualifyingResult(
                    driver_id='verstappen',
                    constructor_id='red_bull',
                    position=verstappen_position,
                    q1_time=LapTime(91100, 1),
                    q2_time=LapTime(90600, 1),
                    q3_time=LapTime(90100, 1)
                ),
                QualifyingResult(
                    driver_id='leclerc',
                    constructor_id='ferrari',
                    position=5,
                    q1_time=LapTime(91200, 1),
                    q2_time=LapTime(90700, 1),
                    q3_time=LapTime(90200, 1)
                )
            ]
            
            race = RaceData(
                season=2024,
                round=i + 1,
                circuit_id=f'circuit_{i}',
                race_name=f'Race {i + 1}',
                date=base_date + timedelta(weeks=i * 2),
                results=results,
                qualifying=qualifying,
                weather=weather
            )
            races.append(race)
        
        return races
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.default_window == 3
        assert calculator.min_races_for_stats == 3
        assert len(calculator.supported_stats) > 0
        assert calculator._stats_cache == {}
    
    def test_calculate_driver_form_success(self, calculator, sample_race_series):
        """Test successful driver form calculation."""
        form_stats = calculator.calculate_driver_form(sample_race_series, 'hamilton', window=3)
        
        assert 'position_stats' in form_stats
        assert 'points_stats' in form_stats
        assert 'consistency_stats' in form_stats
        assert 'form_trends' in form_stats
        
        # Check that we have statistics for multiple time points
        assert len(form_stats['position_stats']) > 0
        assert len(form_stats['points_stats']) > 0
        
        # Check structure of position stats
        position_stat = form_stats['position_stats'][0]
        assert 'position_mean' in position_stat
        assert 'position_std' in position_stat
        assert 'race_date' in position_stat
        assert 'window_size' in position_stat
        assert 'finish_rate' in position_stat
        
        # Check structure of points stats
        points_stat = form_stats['points_stats'][0]
        assert 'points_mean' in points_stat
        assert 'points_per_race' in points_stat
        assert 'total_points' in points_stat
    
    def test_calculate_driver_form_insufficient_data(self, calculator):
        """Test driver form calculation with insufficient data."""
        # Create only 1 race (less than min_races_for_stats)
        race = RaceData(
            season=2024,
            round=1,
            circuit_id='test',
            race_name='Test Race',
            date=datetime(2024, 3, 1),
            results=[
                RaceResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    grid_position=1,
                    final_position=1,
                    points=25,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                )
            ],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        form_stats = calculator.calculate_driver_form([race], 'hamilton')
        assert form_stats == {}
    
    def test_calculate_driver_form_driver_not_found(self, calculator, sample_race_series):
        """Test driver form calculation when driver not found."""
        form_stats = calculator.calculate_driver_form(sample_race_series, 'nonexistent_driver')
        assert form_stats == {}
    
    def test_calculate_driver_form_empty_races(self, calculator):
        """Test driver form calculation with empty race list."""
        form_stats = calculator.calculate_driver_form([], 'hamilton')
        assert form_stats == {}
    
    def test_calculate_constructor_performance_success(self, calculator, sample_race_series):
        """Test successful constructor performance calculation."""
        perf_stats = calculator.calculate_constructor_performance(sample_race_series, 'mercedes', window=3)
        
        assert 'performance_stats' in perf_stats
        assert 'reliability_stats' in perf_stats
        assert 'competitiveness_stats' in perf_stats
        
        # Check performance stats structure
        if perf_stats['performance_stats']:
            perf_stat = perf_stats['performance_stats'][0]
            assert 'avg_points_per_race' in perf_stat
            assert 'total_points' in perf_stat
            assert 'best_avg_position' in perf_stat
            assert 'race_date' in perf_stat
        
        # Check reliability stats structure
        if perf_stats['reliability_stats']:
            reliability_stat = perf_stats['reliability_stats'][0]
            assert 'reliability_rate' in reliability_stat
            assert 'avg_finishers_per_race' in reliability_stat
    
    def test_calculate_constructor_performance_insufficient_data(self, calculator):
        """Test constructor performance with insufficient data."""
        race = RaceData(
            season=2024,
            round=1,
            circuit_id='test',
            race_name='Test Race',
            date=datetime(2024, 3, 1),
            results=[
                RaceResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    grid_position=1,
                    final_position=1,
                    points=25,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                )
            ],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        perf_stats = calculator.calculate_constructor_performance([race], 'mercedes')
        assert perf_stats == {}
    
    def test_calculate_track_specific_form_driver(self, calculator, sample_race_series):
        """Test track-specific form calculation for driver."""
        # Use first circuit
        circuit_id = sample_race_series[0].circuit_id
        
        track_stats = calculator.calculate_track_specific_form(
            sample_race_series, 'hamilton', 'driver', circuit_id
        )
        
        assert 'races_at_track' in track_stats
        assert 'avg_position' in track_stats
        assert 'best_position' in track_stats
        assert 'avg_points' in track_stats
        assert 'finish_rate' in track_stats
        assert 'podium_rate' in track_stats
        
        # Should have exactly 1 race at this specific circuit
        assert track_stats['races_at_track'] == 1
    
    def test_calculate_track_specific_form_constructor(self, calculator, sample_race_series):
        """Test track-specific form calculation for constructor."""
        circuit_id = sample_race_series[0].circuit_id
        
        track_stats = calculator.calculate_track_specific_form(
            sample_race_series, 'mercedes', 'constructor', circuit_id
        )
        
        assert 'races_at_track' in track_stats
        assert 'avg_points_per_race' in track_stats
        assert 'total_points' in track_stats
        assert 'avg_best_position' in track_stats
        assert 'reliability_rate' in track_stats
        assert 'podium_rate' in track_stats
    
    def test_calculate_track_specific_form_all_circuits(self, calculator, sample_race_series):
        """Test track-specific form calculation across all circuits."""
        track_stats = calculator.calculate_track_specific_form(
            sample_race_series, 'hamilton', 'driver', circuit_id=None
        )
        
        # Should include all races
        assert track_stats['races_at_track'] == 6
    
    def test_calculate_track_specific_form_invalid_entity_type(self, calculator, sample_race_series):
        """Test track-specific form with invalid entity type."""
        with pytest.raises(RollingStatsError):
            calculator.calculate_track_specific_form(
                sample_race_series, 'hamilton', 'invalid_type'
            )
    
    def test_calculate_head_to_head_stats_success(self, calculator, sample_race_series):
        """Test successful head-to-head statistics calculation."""
        h2h_stats = calculator.calculate_head_to_head_stats(
            sample_race_series, 'hamilton', 'verstappen', window=3
        )
        
        assert 'head_to_head_rolling' in h2h_stats
        assert 'overall_stats' in h2h_stats
        
        # Check rolling stats structure
        if h2h_stats['head_to_head_rolling']:
            rolling_stat = h2h_stats['head_to_head_rolling'][0]
            assert 'driver1_wins' in rolling_stat
            assert 'driver2_wins' in rolling_stat
            assert 'driver1_win_rate' in rolling_stat
            assert 'driver2_win_rate' in rolling_stat
            assert 'driver1_points_total' in rolling_stat
            assert 'driver2_points_total' in rolling_stat
            assert 'race_date' in rolling_stat
        
        # Check overall stats structure
        overall = h2h_stats['overall_stats']
        assert 'total_races' in overall
        assert 'driver1_wins' in overall
        assert 'driver2_wins' in overall
        assert 'driver1_win_percentage' in overall
        assert 'driver2_win_percentage' in overall
        assert 'total_points_driver1' in overall
        assert 'total_points_driver2' in overall
    
    def test_calculate_head_to_head_stats_insufficient_data(self, calculator):
        """Test head-to-head stats with insufficient data."""
        race = RaceData(
            season=2024,
            round=1,
            circuit_id='test',
            race_name='Test Race',
            date=datetime(2024, 3, 1),
            results=[
                RaceResult(
                    driver_id='hamilton',
                    constructor_id='mercedes',
                    grid_position=1,
                    final_position=1,
                    points=25,
                    fastest_lap=None,
                    status=RaceStatus.FINISHED,
                    laps_completed=57
                )
            ],
            qualifying=[],
            weather=WeatherData(25.0, 60.0, 1013.25, 5.0, 180, False, 30.0)
        )
        
        h2h_stats = calculator.calculate_head_to_head_stats([race], 'hamilton', 'verstappen')
        assert h2h_stats == {}
    
    def test_calculate_window_stats(self, calculator):
        """Test window statistics calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = calculator._calculate_window_stats(values, 'test')
        
        assert 'test_mean' in stats
        assert 'test_median' in stats
        assert 'test_std' in stats
        assert 'test_min' in stats
        assert 'test_max' in stats
        assert 'test_range' in stats
        
        assert stats['test_mean'] == 3.0
        assert stats['test_median'] == 3.0
        assert stats['test_min'] == 1.0
        assert stats['test_max'] == 5.0
        assert stats['test_range'] == 4.0
        
        # Check percentiles (should be present with 5+ values)
        assert 'test_q25' in stats
        assert 'test_q75' in stats
        assert 'test_iqr' in stats
        
        # Check trend (should be present with 3+ values)
        assert 'test_trend' in stats
        assert stats['test_trend'] > 0  # Positive trend
    
    def test_calculate_window_stats_empty(self, calculator):
        """Test window statistics with empty values."""
        stats = calculator._calculate_window_stats([], 'test')
        assert stats == {}
    
    def test_calculate_window_stats_single_value(self, calculator):
        """Test window statistics with single value."""
        stats = calculator._calculate_window_stats([5.0], 'test')
        
        assert stats['test_mean'] == 5.0
        assert stats['test_std'] == 0.0  # No variation
        assert 'test_q25' not in stats  # Not enough values for percentiles
        assert 'test_trend' not in stats  # Not enough values for trend
    
    def test_calculate_consistency_metrics(self, calculator):
        """Test consistency metrics calculation."""
        results = [
            {'final_position': 1, 'finished': True, 'points': 25},
            {'final_position': 2, 'finished': True, 'points': 18},
            {'final_position': 3, 'finished': True, 'points': 15},
            {'final_position': 999, 'finished': False, 'points': 0}  # DNF
        ]
        
        consistency = calculator._calculate_consistency_metrics(results)
        
        assert 'position_consistency' in consistency
        assert 'finish_consistency' in consistency
        assert 'points_cv' in consistency
        
        assert 0 < consistency['position_consistency'] <= 1
        assert consistency['finish_consistency'] == 0.75  # 3 out of 4 finished
        assert consistency['points_cv'] > 0  # Some variation in points
    
    def test_calculate_trend_metrics(self, calculator):
        """Test trend metrics calculation."""
        # Improving performance (positions getting better)
        results = [
            {'final_position': 5, 'finished': True, 'points': 10},
            {'final_position': 3, 'finished': True, 'points': 15},
            {'final_position': 1, 'finished': True, 'points': 25}
        ]
        
        trends = calculator._calculate_trend_metrics(results)
        
        assert 'position_trend' in trends
        assert 'position_momentum' in trends
        assert 'points_trend' in trends
        assert 'points_momentum' in trends
        
        # Position trend should be positive (improving)
        assert trends['position_trend'] > 0
        # Points trend should be positive (increasing)
        assert trends['points_trend'] > 0
        # Position momentum should be positive (recent better than early)
        assert trends['position_momentum'] > 0
        # Points momentum should be positive (recent higher than early)
        assert trends['points_momentum'] > 0
    
    def test_calculate_trend_metrics_insufficient_data(self, calculator):
        """Test trend metrics with insufficient data."""
        results = [
            {'final_position': 1, 'finished': True, 'points': 25},
            {'final_position': 2, 'finished': True, 'points': 18}
        ]
        
        trends = calculator._calculate_trend_metrics(results)
        assert trends == {}  # Not enough data for trends
    
    def test_clear_cache(self, calculator):
        """Test cache clearing functionality."""
        # Add something to cache
        calculator._stats_cache['test'] = 'data'
        calculator._cache_timestamps['test'] = datetime.now()
        
        assert len(calculator._stats_cache) == 1
        assert len(calculator._cache_timestamps) == 1
        
        calculator.clear_cache()
        
        assert len(calculator._stats_cache) == 0
        assert len(calculator._cache_timestamps) == 0
    
    def test_error_handling(self, calculator):
        """Test error handling in rolling statistics calculation."""
        # Test with invalid data that should raise RollingStatsError
        with pytest.raises(RollingStatsError):
            # This should fail due to invalid race data structure
            calculator.calculate_driver_form([None], 'hamilton')