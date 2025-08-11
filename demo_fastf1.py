"""
Demo script for the FastF1 client.
"""
import logging
from src.data.fastf1_client import FastF1Client

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate FastF1 client functionality."""
    client = FastF1Client()
    
    try:
        logger.info("Getting session info for 2023 Bahrain GP...")
        session_info = client.get_session_info(2023, 1, 'R')
        
        logger.info(f"Session: {session_info['session_name']}")
        logger.info(f"Circuit: {session_info['circuit_name']}")
        logger.info(f"Country: {session_info['country']}")
        logger.info(f"Number of drivers: {len(session_info['drivers'])}")
        
        logger.info("\nGetting weather data...")
        weather = client.get_weather_data(2023, 1, 'R')
        
        logger.info(f"Air Temperature: {weather.temperature}°C")
        logger.info(f"Track Temperature: {weather.track_temp}°C")
        logger.info(f"Humidity: {weather.humidity}%")
        logger.info(f"Wind Speed: {weather.wind_speed} km/h")
        logger.info(f"Rainfall: {'Yes' if weather.rainfall else 'No'}")
        
        logger.info("\nGetting available sessions...")
        available_sessions = client.get_available_sessions(2023, 1)
        logger.info(f"Available sessions: {', '.join(available_sessions)}")
        
        logger.info("\nGetting qualifying times for top drivers...")
        try:
            qualifying_times = client.get_qualifying_times(2023, 1)
            
            # Show a few drivers' qualifying times
            for driver_id in list(qualifying_times.keys())[:3]:
                times = qualifying_times[driver_id]
                logger.info(f"{driver_id.upper()}:")
                if times['Q1']:
                    logger.info(f"  Q1: {times['Q1'].time_ms / 1000:.3f}s")
                if times['Q2']:
                    logger.info(f"  Q2: {times['Q2'].time_ms / 1000:.3f}s")
                if times['Q3']:
                    logger.info(f"  Q3: {times['Q3'].time_ms / 1000:.3f}s")
        except Exception as e:
            logger.warning(f"Could not get qualifying times: {e}")
        
        # Validate session data
        is_valid = client.validate_session_data(2023, 1, 'R')
        logger.info(f"\nSession data validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()