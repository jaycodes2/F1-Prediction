"""
Demo script for the combined data collector.
"""
import logging
from src.data.combined_collector import CombinedDataCollector

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate combined data collector functionality."""
    collector = CombinedDataCollector()
    
    try:
        logger.info("Checking data source availability for 2023 Bahrain GP...")
        source_info = collector.get_data_source_info(2023, 1)
        
        logger.info(f"Ergast available: {source_info['ergast_available']}")
        logger.info(f"FastF1 available: {source_info['fastf1_available']}")
        if source_info['fastf1_sessions']:
            logger.info(f"FastF1 sessions: {', '.join(source_info['fastf1_sessions'])}")
        
        logger.info("\nCollecting combined race data...")
        race_data = collector.collect_race_data(2023, 1)
        
        logger.info(f"Race: {race_data.race_name}")
        logger.info(f"Date: {race_data.date}")
        logger.info(f"Circuit: {race_data.circuit_id}")
        logger.info(f"Number of drivers: {len(race_data.results)}")
        
        # Show enhanced weather data
        logger.info(f"\nEnhanced Weather Data:")
        logger.info(f"Air Temperature: {race_data.weather.temperature}°C")
        logger.info(f"Track Temperature: {race_data.weather.track_temp}°C")
        logger.info(f"Humidity: {race_data.weather.humidity}%")
        logger.info(f"Wind Speed: {race_data.weather.wind_speed} km/h")
        logger.info(f"Rainfall: {'Yes' if race_data.weather.rainfall else 'No'}")
        
        # Show top 3 finishers with enhanced data
        logger.info("\nTop 3 finishers (with enhanced data):")
        top_3 = sorted(race_data.results, key=lambda x: x.final_position)[:3]
        for result in top_3:
            fastest_lap_info = ""
            if result.fastest_lap:
                fastest_lap_info = f" (Fastest: {result.fastest_lap.time_ms / 1000:.3f}s)"
            
            logger.info(f"  {result.final_position}. {result.driver_id} ({result.constructor_id}) - {result.points} points{fastest_lap_info}")
        
        # Show enhanced qualifying data
        logger.info("\nTop 3 qualifiers (with enhanced times):")
        top_3_qual = sorted(race_data.qualifying, key=lambda x: x.position)[:3]
        for qual in top_3_qual:
            times_info = []
            if qual.q1_time:
                times_info.append(f"Q1: {qual.q1_time.time_ms / 1000:.3f}s")
            if qual.q2_time:
                times_info.append(f"Q2: {qual.q2_time.time_ms / 1000:.3f}s")
            if qual.q3_time:
                times_info.append(f"Q3: {qual.q3_time.time_ms / 1000:.3f}s")
            
            times_str = ", ".join(times_info) if times_info else "No times available"
            logger.info(f"  {qual.position}. {qual.driver_id} - {times_str}")
        
        # Validate combined data
        is_valid = collector.validate_data(race_data)
        logger.info(f"\nCombined data validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Show data enhancement statistics
        enhanced_weather = race_data.weather.temperature != 25.0  # Default temp
        enhanced_qualifying = any(q.q1_time or q.q2_time or q.q3_time for q in race_data.qualifying)
        enhanced_fastest_laps = sum(1 for r in race_data.results if r.fastest_lap)
        
        logger.info(f"\nData Enhancement Summary:")
        logger.info(f"Weather enhanced: {'Yes' if enhanced_weather else 'No'}")
        logger.info(f"Qualifying times enhanced: {'Yes' if enhanced_qualifying else 'No'}")
        logger.info(f"Fastest laps enhanced: {enhanced_fastest_laps}/{len(race_data.results)} drivers")
        
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()