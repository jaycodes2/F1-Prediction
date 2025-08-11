"""
Rolling statistics calculator for F1 race prediction.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..models.data_models import RaceData, RaceResult, QualifyingResult
from ..config import config


logger = logging.getLogger(__name__)


class RollingStatsError(Exception):
    """Custom exception for rolling statistics errors."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class RollingStatsCalculator:
    """
    Advanced rolling statistics calculator for F1 data.
    Provides time-windowed performance metrics for drivers and constructors.
    """
    
    def __init__(self, default_window: int = None):
        self.default_window = default_window or config.data.rolling_window_size
        self.min_races_for_stats = config.data.min_races_for_features
        
        # Cache for computed statistics
        self._stats_cache = {}
        self._cache_timestamps = {}
        
        # Supported statistics
        self.supported_stats = [
            'mean', 'median', 'std', 'min', 'max', 'trend', 'momentum',
            'consistency', 'volatility', 'percentile_25', 'percentile_75'
        ]
    
    def calculate_driver_form(self, races: List[RaceData], driver_id: str, 
                            window: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate rolling form statistics for a specific driver.
        
        Args:
            races: List of race data (chronologically ordered)
            driver_id: Driver identifier
            window: Rolling window size (default: config value)
            
        Returns:
            Dictionary with rolling statistics over time
        """
        try:
            window = window or self.default_window
            
            if not races:
                return {}
            
            # Sort races by date
            sorted_races = sorted(races, key=lambda x: x.date)
            
            # Extract driver's results
            driver_results = []
            for race in sorted_races:
                for result in race.results:
                    if result.driver_id == driver_id:
                        driver_results.append({
                            'race_date': race.date,
                            'season': race.season,
                            'round': race.round,
                            'circuit_id': race.circuit_id,
                            'grid_position': result.grid_position,
                            'final_position': result.final_position,
                            'points': result.points,
                            'finished': result.final_position < 999,
                            'constructor_id': result.constructor_id,
                            'laps_completed': result.laps_completed
                        })
                        break
            
            if len(driver_results) < self.min_races_for_stats:
                logger.warning(f"Insufficient data for {driver_id}: {len(driver_results)} races")
                return {}
            
            # Calculate rolling statistics
            rolling_stats = {
                'position_stats': [],
                'points_stats': [],
                'consistency_stats': [],
                'form_trends': []
            }
            
            for i in range(len(driver_results)):
                # Define window
                start_idx = max(0, i - window + 1)
                window_results = driver_results[start_idx:i + 1]
                
                if len(window_results) < 2:
                    continue
                
                # Calculate position-based statistics
                positions = [r['final_position'] for r in window_results if r['finished']]
                if positions:
                    position_stats = self._calculate_window_stats(positions, 'position')
                    position_stats.update({
                        'race_date': driver_results[i]['race_date'],
                        'window_size': len(window_results),
                        'finished_races': len(positions),
                        'finish_rate': len(positions) / len(window_results)
                    })
                    rolling_stats['position_stats'].append(position_stats)
                
                # Calculate points-based statistics
                points = [r['points'] for r in window_results]
                points_stats = self._calculate_window_stats(points, 'points')
                points_stats.update({
                    'race_date': driver_results[i]['race_date'],
                    'window_size': len(window_results),
                    'total_points': sum(points),
                    'points_per_race': np.mean(points)
                })
                rolling_stats['points_stats'].append(points_stats)
                
                # Calculate consistency metrics
                if positions:
                    consistency_stats = self._calculate_consistency_metrics(window_results)
                    consistency_stats['race_date'] = driver_results[i]['race_date']
                    rolling_stats['consistency_stats'].append(consistency_stats)
                
                # Calculate form trends
                if len(window_results) >= 3:
                    trend_stats = self._calculate_trend_metrics(window_results)
                    trend_stats['race_date'] = driver_results[i]['race_date']
                    rolling_stats['form_trends'].append(trend_stats)
            
            return rolling_stats
            
        except Exception as e:
            logger.error(f"Error calculating driver form for {driver_id}: {e}")
            raise RollingStatsError(f"Failed to calculate driver form: {e}", 
                                  {"driver_id": driver_id, "window": window})
    
    def calculate_constructor_performance(self, races: List[RaceData], constructor_id: str,
                                        window: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate rolling performance statistics for a constructor.
        
        Args:
            races: List of race data
            constructor_id: Constructor identifier
            window: Rolling window size
            
        Returns:
            Dictionary with constructor rolling statistics
        """
        try:
            window = window or self.default_window
            
            if not races:
                return {}
            
            sorted_races = sorted(races, key=lambda x: x.date)
            
            # Extract constructor results (both drivers)
            constructor_results = []
            for race in sorted_races:
                race_results = []
                for result in race.results:
                    if result.constructor_id == constructor_id:
                        race_results.append({
                            'driver_id': result.driver_id,
                            'grid_position': result.grid_position,
                            'final_position': result.final_position,
                            'points': result.points,
                            'finished': result.final_position < 999
                        })
                
                if race_results:
                    constructor_results.append({
                        'race_date': race.date,
                        'season': race.season,
                        'round': race.round,
                        'circuit_id': race.circuit_id,
                        'drivers': race_results,
                        'total_points': sum(r['points'] for r in race_results),
                        'best_position': min(r['final_position'] for r in race_results if r['finished']) if any(r['finished'] for r in race_results) else 999,
                        'drivers_finished': sum(1 for r in race_results if r['finished']),
                        'total_drivers': len(race_results)
                    })
            
            if len(constructor_results) < self.min_races_for_stats:
                return {}
            
            # Calculate rolling statistics
            rolling_stats = {
                'performance_stats': [],
                'reliability_stats': [],
                'competitiveness_stats': []
            }
            
            for i in range(len(constructor_results)):
                start_idx = max(0, i - window + 1)
                window_results = constructor_results[start_idx:i + 1]
                
                if len(window_results) < 2:
                    continue
                
                # Performance statistics
                total_points = [r['total_points'] for r in window_results]
                best_positions = [r['best_position'] for r in window_results if r['best_position'] < 999]
                
                perf_stats = {
                    'race_date': constructor_results[i]['race_date'],
                    'window_size': len(window_results),
                    'avg_points_per_race': np.mean(total_points),
                    'total_points': sum(total_points),
                    'points_consistency': np.std(total_points) if len(total_points) > 1 else 0,
                    'best_avg_position': np.mean(best_positions) if best_positions else 20,
                    'podium_races': sum(1 for pos in best_positions if pos <= 3),
                    'points_scoring_races': sum(1 for points in total_points if points > 0)
                }
                rolling_stats['performance_stats'].append(perf_stats)
                
                # Reliability statistics
                total_driver_races = sum(r['total_drivers'] for r in window_results)
                total_finishes = sum(r['drivers_finished'] for r in window_results)
                
                reliability_stats = {
                    'race_date': constructor_results[i]['race_date'],
                    'window_size': len(window_results),
                    'reliability_rate': total_finishes / total_driver_races if total_driver_races > 0 else 0,
                    'avg_finishers_per_race': total_finishes / len(window_results),
                    'both_cars_finished': sum(1 for r in window_results if r['drivers_finished'] == r['total_drivers']),
                    'double_dnf': sum(1 for r in window_results if r['drivers_finished'] == 0)
                }
                rolling_stats['reliability_stats'].append(reliability_stats)
                
                # Competitiveness (relative to field)
                competitiveness_stats = self._calculate_constructor_competitiveness(window_results)
                competitiveness_stats['race_date'] = constructor_results[i]['race_date']
                rolling_stats['competitiveness_stats'].append(competitiveness_stats)
            
            return rolling_stats
            
        except Exception as e:
            logger.error(f"Error calculating constructor performance for {constructor_id}: {e}")
            raise RollingStatsError(f"Failed to calculate constructor performance: {e}",
                                  {"constructor_id": constructor_id, "window": window})
    
    def calculate_track_specific_form(self, races: List[RaceData], entity_id: str,
                                    entity_type: str = 'driver', circuit_id: str = None) -> Dict[str, Any]:
        """
        Calculate track-specific rolling statistics.
        
        Args:
            races: List of race data
            entity_id: Driver or constructor ID
            entity_type: 'driver' or 'constructor'
            circuit_id: Specific circuit (None for all circuits)
            
        Returns:
            Dictionary with track-specific statistics
        """
        try:
            if not races:
                return {}
            
            # Filter races by circuit if specified
            if circuit_id:
                filtered_races = [race for race in races if race.circuit_id == circuit_id]
            else:
                filtered_races = races
            
            if not filtered_races:
                return {}
            
            sorted_races = sorted(filtered_races, key=lambda x: x.date)
            
            if entity_type == 'driver':
                return self._calculate_driver_track_stats(sorted_races, entity_id, circuit_id)
            elif entity_type == 'constructor':
                return self._calculate_constructor_track_stats(sorted_races, entity_id, circuit_id)
            else:
                raise ValueError(f"Invalid entity_type: {entity_type}")
                
        except Exception as e:
            logger.error(f"Error calculating track-specific form: {e}")
            raise RollingStatsError(f"Failed to calculate track-specific form: {e}")
    
    def calculate_head_to_head_stats(self, races: List[RaceData], driver1_id: str, driver2_id: str,
                                   window: int = None) -> Dict[str, Any]:
        """
        Calculate head-to-head rolling statistics between two drivers.
        
        Args:
            races: List of race data
            driver1_id: First driver ID
            driver2_id: Second driver ID
            window: Rolling window size
            
        Returns:
            Dictionary with head-to-head statistics
        """
        try:
            window = window or self.default_window
            
            if not races:
                return {}
            
            sorted_races = sorted(races, key=lambda x: x.date)
            
            # Find races where both drivers participated
            head_to_head_results = []
            for race in sorted_races:
                driver1_result = None
                driver2_result = None
                
                for result in race.results:
                    if result.driver_id == driver1_id:
                        driver1_result = result
                    elif result.driver_id == driver2_id:
                        driver2_result = result
                
                if driver1_result and driver2_result:
                    head_to_head_results.append({
                        'race_date': race.date,
                        'circuit_id': race.circuit_id,
                        'driver1_position': driver1_result.final_position,
                        'driver2_position': driver2_result.final_position,
                        'driver1_points': driver1_result.points,
                        'driver2_points': driver2_result.points,
                        'driver1_grid': driver1_result.grid_position,
                        'driver2_grid': driver2_result.grid_position,
                        'driver1_finished': driver1_result.final_position < 999,
                        'driver2_finished': driver2_result.final_position < 999
                    })
            
            if len(head_to_head_results) < 2:
                return {}
            
            # Calculate rolling head-to-head statistics
            rolling_h2h = []
            for i in range(len(head_to_head_results)):
                start_idx = max(0, i - window + 1)
                window_results = head_to_head_results[start_idx:i + 1]
                
                if len(window_results) < 2:
                    continue
                
                # Calculate head-to-head metrics
                driver1_wins = 0
                driver2_wins = 0
                driver1_points_total = 0
                driver2_points_total = 0
                both_finished = 0
                
                for result in window_results:
                    if result['driver1_finished'] and result['driver2_finished']:
                        both_finished += 1
                        if result['driver1_position'] < result['driver2_position']:
                            driver1_wins += 1
                        elif result['driver2_position'] < result['driver1_position']:
                            driver2_wins += 1
                    
                    driver1_points_total += result['driver1_points']
                    driver2_points_total += result['driver2_points']
                
                h2h_stats = {
                    'race_date': head_to_head_results[i]['race_date'],
                    'window_size': len(window_results),
                    'races_both_finished': both_finished,
                    'driver1_wins': driver1_wins,
                    'driver2_wins': driver2_wins,
                    'driver1_win_rate': driver1_wins / both_finished if both_finished > 0 else 0,
                    'driver2_win_rate': driver2_wins / both_finished if both_finished > 0 else 0,
                    'driver1_points_total': driver1_points_total,
                    'driver2_points_total': driver2_points_total,
                    'driver1_avg_points': driver1_points_total / len(window_results),
                    'driver2_avg_points': driver2_points_total / len(window_results),
                    'points_advantage_driver1': driver1_points_total - driver2_points_total
                }
                
                rolling_h2h.append(h2h_stats)
            
            return {
                'head_to_head_rolling': rolling_h2h,
                'overall_stats': self._calculate_overall_h2h_stats(head_to_head_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating head-to-head stats: {e}")
            raise RollingStatsError(f"Failed to calculate head-to-head stats: {e}")
    
    def _calculate_window_stats(self, values: List[float], stat_type: str) -> Dict[str, float]:
        """Calculate statistical measures for a window of values."""
        if not values:
            return {}
        
        values_array = np.array(values)
        
        stats = {
            f'{stat_type}_mean': np.mean(values_array),
            f'{stat_type}_median': np.median(values_array),
            f'{stat_type}_std': np.std(values_array) if len(values) > 1 else 0,
            f'{stat_type}_min': np.min(values_array),
            f'{stat_type}_max': np.max(values_array),
            f'{stat_type}_range': np.max(values_array) - np.min(values_array)
        }
        
        # Add percentiles
        if len(values) >= 4:
            stats[f'{stat_type}_q25'] = np.percentile(values_array, 25)
            stats[f'{stat_type}_q75'] = np.percentile(values_array, 75)
            stats[f'{stat_type}_iqr'] = stats[f'{stat_type}_q75'] - stats[f'{stat_type}_q25']
        
        # Add trend if enough data
        if len(values) >= 3:
            x = np.arange(len(values))
            slope = np.polyfit(x, values_array, 1)[0]
            stats[f'{stat_type}_trend'] = slope
        
        return stats
    
    def _calculate_consistency_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate consistency metrics for a window of results."""
        positions = [r['final_position'] for r in results if r['finished']]
        points = [r['points'] for r in results]
        
        consistency_metrics = {}
        
        if positions:
            # Position consistency (lower std is better)
            position_std = np.std(positions)
            consistency_metrics['position_consistency'] = 1 / (1 + position_std)  # Normalized
            
            # Finishing consistency
            finish_rate = len(positions) / len(results)
            consistency_metrics['finish_consistency'] = finish_rate
        
        if points:
            # Points consistency
            points_std = np.std(points)
            points_mean = np.mean(points)
            if points_mean > 0:
                consistency_metrics['points_cv'] = points_std / points_mean
            else:
                consistency_metrics['points_cv'] = float('inf')
        
        return consistency_metrics
    
    def _calculate_trend_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend metrics for a window of results."""
        positions = [r['final_position'] for r in results if r['finished']]
        points = [r['points'] for r in results]
        
        trend_metrics = {}
        
        if len(positions) >= 3:
            # Position trend (negative slope means improving positions)
            x = np.arange(len(positions))
            position_slope = np.polyfit(x, positions, 1)[0]
            trend_metrics['position_trend'] = -position_slope  # Invert so positive is better
            
            # Momentum (recent vs early performance)
            recent_positions = positions[-2:] if len(positions) >= 4 else positions[-1:]
            early_positions = positions[:2] if len(positions) >= 4 else positions[:1]
            trend_metrics['position_momentum'] = np.mean(early_positions) - np.mean(recent_positions)
        
        if len(points) >= 3:
            # Points trend
            x = np.arange(len(points))
            points_slope = np.polyfit(x, points, 1)[0]
            trend_metrics['points_trend'] = points_slope
            
            # Points momentum
            recent_points = points[-2:] if len(points) >= 4 else points[-1:]
            early_points = points[:2] if len(points) >= 4 else points[:1]
            trend_metrics['points_momentum'] = np.mean(recent_points) - np.mean(early_points)
        
        return trend_metrics
    
    def _calculate_constructor_competitiveness(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate constructor competitiveness metrics."""
        # This is a simplified version - in practice, you'd compare against field average
        total_points = [r['total_points'] for r in results]
        best_positions = [r['best_position'] for r in results if r['best_position'] < 999]
        
        competitiveness = {}
        
        if total_points:
            competitiveness['avg_points_per_race'] = np.mean(total_points)
            competitiveness['points_trend'] = np.polyfit(range(len(total_points)), total_points, 1)[0] if len(total_points) >= 3 else 0
        
        if best_positions:
            competitiveness['avg_best_position'] = np.mean(best_positions)
            competitiveness['top5_rate'] = sum(1 for pos in best_positions if pos <= 5) / len(best_positions)
            competitiveness['podium_rate'] = sum(1 for pos in best_positions if pos <= 3) / len(best_positions)
        
        return competitiveness
    
    def _calculate_driver_track_stats(self, races: List[RaceData], driver_id: str, circuit_id: str) -> Dict[str, Any]:
        """Calculate driver-specific track statistics."""
        driver_results = []
        
        for race in races:
            for result in race.results:
                if result.driver_id == driver_id:
                    driver_results.append({
                        'race_date': race.date,
                        'final_position': result.final_position,
                        'grid_position': result.grid_position,
                        'points': result.points,
                        'finished': result.final_position < 999
                    })
                    break
        
        if not driver_results:
            return {}
        
        # Calculate track-specific metrics
        positions = [r['final_position'] for r in driver_results if r['finished']]
        points = [r['points'] for r in driver_results]
        
        track_stats = {
            'races_at_track': len(driver_results),
            'avg_position': np.mean(positions) if positions else 20,
            'best_position': min(positions) if positions else 20,
            'avg_points': np.mean(points),
            'total_points': sum(points),
            'finish_rate': len(positions) / len(driver_results),
            'podium_rate': sum(1 for pos in positions if pos <= 3) / len(driver_results),
            'points_rate': sum(1 for p in points if p > 0) / len(driver_results)
        }
        
        return track_stats
    
    def _calculate_constructor_track_stats(self, races: List[RaceData], constructor_id: str, circuit_id: str) -> Dict[str, Any]:
        """Calculate constructor-specific track statistics."""
        constructor_results = []
        
        for race in races:
            race_points = 0
            race_positions = []
            drivers_finished = 0
            total_drivers = 0
            
            for result in race.results:
                if result.constructor_id == constructor_id:
                    total_drivers += 1
                    race_points += result.points
                    if result.final_position < 999:
                        race_positions.append(result.final_position)
                        drivers_finished += 1
            
            if total_drivers > 0:
                constructor_results.append({
                    'race_date': race.date,
                    'total_points': race_points,
                    'best_position': min(race_positions) if race_positions else 20,
                    'drivers_finished': drivers_finished,
                    'total_drivers': total_drivers
                })
        
        if not constructor_results:
            return {}
        
        # Calculate constructor track-specific metrics
        total_points = [r['total_points'] for r in constructor_results]
        best_positions = [r['best_position'] for r in constructor_results]
        
        track_stats = {
            'races_at_track': len(constructor_results),
            'avg_points_per_race': np.mean(total_points),
            'total_points': sum(total_points),
            'avg_best_position': np.mean(best_positions),
            'best_ever_position': min(best_positions),
            'reliability_rate': sum(r['drivers_finished'] for r in constructor_results) / sum(r['total_drivers'] for r in constructor_results),
            'podium_rate': sum(1 for pos in best_positions if pos <= 3) / len(constructor_results)
        }
        
        return track_stats
    
    def _calculate_overall_h2h_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall head-to-head statistics."""
        if not results:
            return {}
        
        driver1_wins = 0
        driver2_wins = 0
        both_finished_races = 0
        
        for result in results:
            if result['driver1_finished'] and result['driver2_finished']:
                both_finished_races += 1
                if result['driver1_position'] < result['driver2_position']:
                    driver1_wins += 1
                elif result['driver2_position'] < result['driver1_position']:
                    driver2_wins += 1
        
        total_driver1_points = sum(r['driver1_points'] for r in results)
        total_driver2_points = sum(r['driver2_points'] for r in results)
        
        return {
            'total_races': len(results),
            'races_both_finished': both_finished_races,
            'driver1_wins': driver1_wins,
            'driver2_wins': driver2_wins,
            'driver1_win_percentage': (driver1_wins / both_finished_races * 100) if both_finished_races > 0 else 0,
            'driver2_win_percentage': (driver2_wins / both_finished_races * 100) if both_finished_races > 0 else 0,
            'total_points_driver1': total_driver1_points,
            'total_points_driver2': total_driver2_points,
            'points_advantage_driver1': total_driver1_points - total_driver2_points
        }
    
    def clear_cache(self):
        """Clear the statistics cache."""
        self._stats_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Rolling statistics cache cleared")