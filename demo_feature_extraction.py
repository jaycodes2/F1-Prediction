"""
Demo script for the base feature extraction system.
"""
import logging
import pandas as pd
from src.features.base_extractor import BaseFeatureExtractor
from src.data.combined_collector import CombinedDataCollector

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate feature extraction functionality."""
    # Initialize components
    extractor = BaseFeatureExtractor()
    collector = CombinedDataCollector()
    
    try:
        logger.info("=== Feature Extraction Demo ===")
        
        # Collect historical data for feature extraction
        logger.info("\nCollecting historical race data...")
        historical_races = []
        
        # Collect a few races from 2023 for feature extraction
        for round_num in [1, 2, 3]:
            try:
                race_data = collector.collect_race_data(2023, round_num)
                historical_races.append(race_data)
                logger.info(f"Collected 2023 Round {round_num}: {race_data.race_name}")
            except Exception as e:
                logger.warning(f"Could not collect 2023 Round {round_num}: {e}")
                continue
        
        if not historical_races:
            logger.error("No historical data collected. Cannot demonstrate feature extraction.")
            return
        
        # Extract features for a specific driver
        logger.info(f"\nExtracting features for Hamilton from {len(historical_races)} races...")
        try:
            hamilton_features = extractor.extract_driver_features(historical_races, 'hamilton')
            
            logger.info("Hamilton's Features:")
            logger.info(f"  Recent Form: {hamilton_features.recent_form:.3f}")
            logger.info(f"  Constructor Performance: {hamilton_features.constructor_performance:.3f}")
            logger.info(f"  Track Experience: {hamilton_features.track_experience} races")
            logger.info(f"  Weather Performance: {hamilton_features.weather_performance:.3f}")
            logger.info(f"  Qualifying Delta: {hamilton_features.qualifying_delta:.3f}")
            logger.info(f"  Championship Position: {hamilton_features.championship_position}")
            logger.info(f"  Points Total: {hamilton_features.points_total}")
            
        except Exception as e:
            logger.warning(f"Could not extract Hamilton features: {e}")
        
        # Extract features for multiple drivers
        logger.info("\nExtracting features for multiple drivers...")
        drivers_to_analyze = ['hamilton', 'verstappen', 'leclerc', 'russell']
        driver_features_summary = {}
        
        for driver_id in drivers_to_analyze:
            try:
                features = extractor.extract_driver_features(historical_races, driver_id)
                driver_features_summary[driver_id] = {
                    'recent_form': features.recent_form,
                    'constructor_performance': features.constructor_performance,
                    'track_experience': features.track_experience,
                    'points_total': features.points_total
                }
            except Exception as e:
                logger.warning(f"Could not extract features for {driver_id}: {e}")
                continue
        
        # Display comparison
        if driver_features_summary:
            logger.info("\nDriver Features Comparison:")
            logger.info(f"{'Driver':<12} {'Form':<6} {'Constructor':<12} {'Experience':<10} {'Points':<8}")
            logger.info("-" * 60)
            
            for driver_id, features in driver_features_summary.items():
                logger.info(f"{driver_id:<12} {features['recent_form']:<6.3f} "
                          f"{features['constructor_performance']:<12.3f} "
                          f"{features['track_experience']:<10} {features['points_total']:<8.0f}")
        
        # Calculate rolling statistics
        logger.info(f"\nCalculating rolling statistics (window=3)...")
        rolling_stats = extractor.calculate_rolling_stats(historical_races, window=3)
        
        logger.info(f"Rolling stats calculated for:")
        logger.info(f"  Drivers: {len(rolling_stats['driver_form'])} drivers")
        logger.info(f"  Constructors: {len(rolling_stats['constructor_performance'])} constructors")
        
        # Show rolling form for Hamilton
        if 'hamilton' in rolling_stats['driver_form']:
            hamilton_rolling = rolling_stats['driver_form']['hamilton']
            logger.info(f"\nHamilton's Rolling Form:")
            for entry in hamilton_rolling[-3:]:  # Show last 3 entries
                race_date = entry['race_date'].strftime('%Y-%m-%d')
                logger.info(f"  {race_date}: Form = {entry['form']:.3f} "
                          f"(based on {entry['races_in_window']} races)")
        
        # Demonstrate categorical encoding
        logger.info("\nDemonstrating categorical feature encoding...")
        sample_features = {
            'constructor_id': 'mercedes',
            'circuit_id': 'bahrain',
            'weather_condition': 'dry',
            'recent_form': 0.85,
            'track_experience': 15
        }
        
        logger.info("Original features:")
        for key, value in sample_features.items():
            logger.info(f"  {key}: {value}")
        
        encoded_features = extractor.encode_categorical_features(sample_features)
        
        logger.info(f"\nEncoded features ({len(encoded_features)} total):")
        # Show a few examples
        categorical_examples = [k for k in encoded_features.keys() if '_' in k][:5]
        for key in categorical_examples:
            logger.info(f"  {key}: {encoded_features[key]}")
        
        logger.info(f"  ... and {len(encoded_features) - len(categorical_examples)} more features")
        
        # Extract complete race features
        if len(historical_races) >= 2:
            logger.info("\nExtracting complete race features...")
            latest_race = historical_races[-1]
            previous_races = historical_races[:-1]
            
            race_features = extractor.extract_race_features(latest_race, previous_races)
            
            logger.info(f"Race features extracted for: {latest_race.race_name}")
            logger.info(f"Feature categories:")
            for category, features in race_features.items():
                if isinstance(features, dict):
                    logger.info(f"  {category}: {len(features)} features")
                else:
                    logger.info(f"  {category}: {type(features).__name__}")
            
            # Show race metadata features
            metadata = race_features['race_metadata']
            logger.info(f"\nRace Metadata Features:")
            logger.info(f"  Season: {metadata['season']}")
            logger.info(f"  Round: {metadata['round']}")
            logger.info(f"  Circuit: {metadata['circuit_id']}")
            logger.info(f"  Month: {metadata['month']}")
            logger.info(f"  Day of Year: {metadata['day_of_year']}")
            
            # Show weather features
            weather = race_features['weather_features']
            logger.info(f"\nWeather Features:")
            logger.info(f"  Temperature: {weather['temperature']}°C")
            logger.info(f"  Humidity: {weather['humidity']}%")
            logger.info(f"  Condition: {weather['weather_condition']}")
            logger.info(f"  Rainfall: {'Yes' if weather['rainfall'] else 'No'}")
            
            # Show driver features count
            driver_features = race_features['driver_features']
            logger.info(f"\nDriver Features: {len(driver_features)} drivers")
            
            # Show example driver features
            if 'hamilton' in driver_features:
                hamilton_race_features = driver_features['hamilton']
                logger.info(f"Hamilton's race features:")
                for key, value in list(hamilton_race_features.items())[:5]:
                    logger.info(f"  {key}: {value}")
        
        logger.info("\n=== Feature extraction demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main()