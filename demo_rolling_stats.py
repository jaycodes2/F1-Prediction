"""
Demo script for the rolling statistics calculator.
"""
import logging
from src.features.rolling_stats import RollingStatsCalculator
from src.data.combined_collector import CombinedDataCollector

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate rolling statistics functionality."""
    # Initialize components
    calculator = RollingStatsCalculator(default_window=5)
    collector = CombinedDataCollector()
    
    try:
        logger.info("=== Rolling Statistics Demo ===")
        
        # Collect historical data for rolling statistics
        logger.info("\nCollecting historical race data...")
        historical_races = []
        
        # Collect races from 2023 for rolling statistics
        for round_num in range(1, 8):  # First 7 races of 2023
            try:
                race_data = collector.collect_race_data(2023, round_num)
                historical_races.append(race_data)
                logger.info(f"Collected 2023 Round {round_num}: {race_data.race_name}")
            except Exception as e:
                logger.warning(f"Could not collect 2023 Round {round_num}: {e}")
                continue
        
        if len(historical_races) < 3:
            logger.error("Insufficient historical data for rolling statistics demo.")
            return
        
        logger.info(f"\nCollected {len(historical_races)} races for analysis")
        
        # Calculate driver form for Hamilton
        logger.info("\n=== Driver Form Analysis (Hamilton) ===")
        try:
            hamilton_form = calculator.calculate_driver_form(historical_races, 'hamilton', window=3)
            
            if hamilton_form and hamilton_form['position_stats']:
                logger.info("Hamilton's Rolling Position Statistics:")
                for i, stat in enumerate(hamilton_form['position_stats'][-3:]):  # Show last 3
                    race_date = stat['race_date'].strftime('%Y-%m-%d')
                    logger.info(f"  {race_date}: Avg Position = {stat['position_mean']:.1f}, "
                              f"Finish Rate = {stat['finish_rate']:.1%}, "
                              f"Window = {stat['window_size']} races")
                
                logger.info("\nHamilton's Rolling Points Statistics:")
                for i, stat in enumerate(hamilton_form['points_stats'][-3:]):  # Show last 3
                    race_date = stat['race_date'].strftime('%Y-%m-%d')
                    logger.info(f"  {race_date}: Avg Points = {stat['points_per_race']:.1f}, "
                              f"Total = {stat['total_points']:.0f}, "
                              f"Consistency = {stat.get('points_std', 0):.1f}")
                
                if hamilton_form['form_trends']:
                    logger.info("\nHamilton's Form Trends:")
                    latest_trend = hamilton_form['form_trends'][-1]
                    logger.info(f"  Position Trend: {latest_trend.get('position_trend', 0):.3f} "
                              f"(positive = improving)")
                    logger.info(f"  Points Trend: {latest_trend.get('points_trend', 0):.3f} "
                              f"(positive = increasing)")
                    logger.info(f"  Position Momentum: {latest_trend.get('position_momentum', 0):.3f}")
            else:
                logger.warning("No form statistics available for Hamilton")
        
        except Exception as e:
            logger.warning(f"Could not calculate Hamilton's form: {e}")
        
        # Calculate constructor performance for Mercedes
        logger.info("\n=== Constructor Performance Analysis (Mercedes) ===")
        try:
            mercedes_perf = calculator.calculate_constructor_performance(historical_races, 'mercedes', window=3)
            
            if mercedes_perf and mercedes_perf['performance_stats']:
                logger.info("Mercedes Rolling Performance:")
                for stat in mercedes_perf['performance_stats'][-3:]:  # Show last 3
                    race_date = stat['race_date'].strftime('%Y-%m-%d')
                    logger.info(f"  {race_date}: Avg Points = {stat['avg_points_per_race']:.1f}, "
                              f"Best Pos = {stat['best_avg_position']:.1f}, "
                              f"Podiums = {stat['podium_races']}")
                
                logger.info("\nMercedes Reliability:")
                for stat in mercedes_perf['reliability_stats'][-3:]:  # Show last 3
                    race_date = stat['race_date'].strftime('%Y-%m-%d')
                    logger.info(f"  {race_date}: Reliability = {stat['reliability_rate']:.1%}, "
                              f"Avg Finishers = {stat['avg_finishers_per_race']:.1f}, "
                              f"Both Cars Finished = {stat['both_cars_finished']} races")
            else:
                logger.warning("No performance statistics available for Mercedes")
        
        except Exception as e:
            logger.warning(f"Could not calculate Mercedes performance: {e}")
        
        # Head-to-head comparison
        logger.info("\n=== Head-to-Head Analysis (Hamilton vs Verstappen) ===")
        try:
            h2h_stats = calculator.calculate_head_to_head_stats(
                historical_races, 'hamilton', 'verstappen', window=4
            )
            
            if h2h_stats and h2h_stats['overall_stats']:
                overall = h2h_stats['overall_stats']
                logger.info("Overall Head-to-Head Statistics:")
                logger.info(f"  Total Races: {overall['total_races']}")
                logger.info(f"  Races Both Finished: {overall['races_both_finished']}")
                logger.info(f"  Hamilton Wins: {overall['driver1_wins']} "
                          f"({overall['driver1_win_percentage']:.1f}%)")
                logger.info(f"  Verstappen Wins: {overall['driver2_wins']} "
                          f"({overall['driver2_win_percentage']:.1f}%)")
                logger.info(f"  Points - Hamilton: {overall['total_points_driver1']:.0f}, "
                          f"Verstappen: {overall['total_points_driver2']:.0f}")
                logger.info(f"  Points Advantage (Hamilton): {overall['points_advantage_driver1']:.0f}")
                
                if h2h_stats['head_to_head_rolling']:
                    logger.info("\nRolling Head-to-Head (Last 3 Windows):")
                    for stat in h2h_stats['head_to_head_rolling'][-3:]:
                        race_date = stat['race_date'].strftime('%Y-%m-%d')
                        logger.info(f"  {race_date}: HAM {stat['driver1_wins']}-{stat['driver2_wins']} VER "
                                  f"(Win Rate: {stat['driver1_win_rate']:.1%} vs {stat['driver2_win_rate']:.1%})")
            else:
                logger.warning("No head-to-head statistics available")
        
        except Exception as e:
            logger.warning(f"Could not calculate head-to-head stats: {e}")
        
        # Track-specific analysis
        logger.info("\n=== Track-Specific Analysis ===")
        try:
            # Analyze Hamilton's performance at a specific circuit
            if historical_races:
                first_circuit = historical_races[0].circuit_id
                logger.info(f"Analyzing Hamilton's performance at {first_circuit}:")
                
                track_stats = calculator.calculate_track_specific_form(
                    historical_races, 'hamilton', 'driver', first_circuit
                )
                
                if track_stats:
                    logger.info(f"  Races at {first_circuit}: {track_stats['races_at_track']}")
                    logger.info(f"  Average Position: {track_stats['avg_position']:.1f}")
                    logger.info(f"  Best Position: {track_stats['best_position']}")
                    logger.info(f"  Average Points: {track_stats['avg_points']:.1f}")
                    logger.info(f"  Finish Rate: {track_stats['finish_rate']:.1%}")
                    logger.info(f"  Podium Rate: {track_stats['podium_rate']:.1%}")
                
                # Compare with overall performance
                logger.info(f"\nHamilton's overall performance across all circuits:")
                overall_track_stats = calculator.calculate_track_specific_form(
                    historical_races, 'hamilton', 'driver', circuit_id=None
                )
                
                if overall_track_stats:
                    logger.info(f"  Total Races: {overall_track_stats['races_at_track']}")
                    logger.info(f"  Overall Avg Position: {overall_track_stats['avg_position']:.1f}")
                    logger.info(f"  Overall Best Position: {overall_track_stats['best_position']}")
                    logger.info(f"  Overall Finish Rate: {overall_track_stats['finish_rate']:.1%}")
        
        except Exception as e:
            logger.warning(f"Could not calculate track-specific stats: {e}")
        
        # Constructor comparison
        logger.info("\n=== Constructor Comparison ===")
        constructors_to_analyze = ['mercedes', 'red_bull', 'ferrari']
        
        for constructor_id in constructors_to_analyze:
            try:
                constructor_perf = calculator.calculate_constructor_performance(
                    historical_races, constructor_id, window=len(historical_races)
                )
                
                if constructor_perf and constructor_perf['performance_stats']:
                    latest_perf = constructor_perf['performance_stats'][-1]
                    logger.info(f"{constructor_id.upper()}:")
                    logger.info(f"  Avg Points/Race: {latest_perf['avg_points_per_race']:.1f}")
                    logger.info(f"  Best Avg Position: {latest_perf['best_avg_position']:.1f}")
                    logger.info(f"  Podium Races: {latest_perf['podium_races']}")
                    logger.info(f"  Points Scoring Races: {latest_perf['points_scoring_races']}")
            
            except Exception as e:
                logger.warning(f"Could not analyze {constructor_id}: {e}")
        
        logger.info("\n=== Rolling statistics demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main()