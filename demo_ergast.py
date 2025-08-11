"""
Demo script for the Ergast API client.
"""
import logging
from src.data.ergast_client import ErgastAPIClient

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate Ergast API client functionality."""
    client = ErgastAPIClient()
    
    try:
        logger.info("Collecting race data for 2023 Bahrain GP...")
        race_data = client.collect_race_data(2023, 1)
        
        logger.info(f"Race: {race_data.race_name}")
        logger.info(f"Date: {race_data.date}")
        logger.info(f"Circuit: {race_data.circuit_id}")
        logger.info(f"Number of drivers: {len(race_data.results)}")
        
        # Show top 3 finishers
        logger.info("\nTop 3 finishers:")
        top_3 = sorted(race_data.results, key=lambda x: x.final_position)[:3]
        for result in top_3:
            logger.info(f"  {result.final_position}. {result.driver_id} ({result.constructor_id}) - {result.points} points")
        
        # Show qualifying results
        logger.info("\nTop 3 qualifiers:")
        top_3_qual = sorted(race_data.qualifying, key=lambda x: x.position)[:3]
        for qual in top_3_qual:
            q3_time = qual.q3_time.time_ms / 1000 if qual.q3_time else "N/A"
            logger.info(f"  {qual.position}. {qual.driver_id} - Q3: {q3_time}s")
        
        # Validate data
        is_valid = client.validate_data(race_data)
        logger.info(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()