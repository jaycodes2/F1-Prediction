"""
Demo script for the data storage and validation system.
"""
import logging
from datetime import datetime
from src.data.storage import FileDataStorage
from src.data.validator import DataValidator
from src.data.combined_collector import CombinedDataCollector

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate storage and validation functionality."""
    # Initialize components
    storage = FileDataStorage()
    validator = DataValidator()
    collector = CombinedDataCollector()
    
    try:
        logger.info("=== Data Storage and Validation Demo ===")
        
        # Show initial storage stats
        logger.info("\nInitial storage statistics:")
        stats = storage.get_storage_stats()
        logger.info(f"Total races: {stats['total_races']}")
        logger.info(f"Total features: {stats['total_features']}")
        logger.info(f"Storage size: {stats['storage_size_mb']} MB")
        logger.info(f"Seasons: {stats['seasons']}")
        
        # Collect some race data
        logger.info("\nCollecting race data for 2023 Bahrain GP...")
        race_data = collector.collect_race_data(2023, 1)
        
        # Validate the collected data
        logger.info("\nValidating collected race data...")
        validation_result = validator.validate_race_data(race_data)
        
        logger.info(f"Validation result: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        logger.info(f"Errors: {len(validation_result.errors)}")
        logger.info(f"Warnings: {len(validation_result.warnings)}")
        
        if validation_result.errors:
            logger.info("Validation errors:")
            for error in validation_result.errors[:3]:  # Show first 3 errors
                logger.info(f"  - {error.message}")
        
        if validation_result.warnings:
            logger.info("Validation warnings:")
            for warning in validation_result.warnings[:3]:  # Show first 3 warnings
                logger.info(f"  - {warning.message}")
        
        # Show validation statistics
        logger.info(f"\nValidation statistics:")
        for key, value in validation_result.stats.items():
            logger.info(f"  {key}: {value}")
        
        # Save the race data
        logger.info("\nSaving race data to storage...")
        success = storage.save_race_data(race_data)
        logger.info(f"Save result: {'SUCCESS' if success else 'FAILED'}")
        
        # Save some sample features
        logger.info("\nSaving sample features...")
        sample_features = {
            'driver_form_hamilton': [0.9, 0.8, 0.85, 0.92, 0.88],
            'constructor_performance_mercedes': 0.87,
            'track_characteristics': {
                'overtaking_difficulty': 0.6,
                'weather_sensitivity': 0.4,
                'tire_degradation': 0.7
            },
            'feature_timestamp': datetime.now().isoformat()
        }
        
        storage.save_features(sample_features, 'bahrain_2023_features')
        
        # Show updated storage stats
        logger.info("\nUpdated storage statistics:")
        stats = storage.get_storage_stats()
        logger.info(f"Total races: {stats['total_races']}")
        logger.info(f"Total features: {stats['total_features']}")
        logger.info(f"Storage size: {stats['storage_size_mb']} MB")
        logger.info(f"Last updated: {stats['last_updated']}")
        
        # List available races
        logger.info("\nAvailable races in storage:")
        available_races = storage.list_available_races()
        for race in available_races:
            logger.info(f"  {race['season']} Round {race['round']}: {race['race_name']} ({race['num_drivers']} drivers)")
        
        # Test loading data back
        logger.info("\nTesting data retrieval...")
        loaded_race_data = storage.load_race_data(2023, 1)
        if loaded_race_data:
            logger.info(f"Successfully loaded: {loaded_race_data.race_name}")
            logger.info(f"Date: {loaded_race_data.date}")
            logger.info(f"Drivers: {len(loaded_race_data.results)}")
        
        loaded_features = storage.load_features('bahrain_2023_features')
        if loaded_features:
            logger.info(f"Successfully loaded features with {len(loaded_features)} items")
        
        # Validate the loaded data
        logger.info("\nValidating loaded data...")
        if loaded_race_data:
            loaded_validation = validator.validate_race_data(loaded_race_data)
            logger.info(f"Loaded data validation: {'PASSED' if loaded_validation.is_valid else 'FAILED'}")
        
        # Demonstrate multiple race validation
        logger.info("\nCollecting additional race data for batch validation...")
        try:
            race_data_2 = collector.collect_race_data(2023, 2)
            storage.save_race_data(race_data_2)
            
            # Validate multiple races
            races = [race_data, race_data_2]
            aggregate_validation = validator.validate_multiple_races(races)
            
            logger.info(f"\nBatch validation results:")
            logger.info(f"Total races: {aggregate_validation['total_races']}")
            logger.info(f"Valid races: {aggregate_validation['valid_races']}")
            logger.info(f"Total errors: {aggregate_validation['total_errors']}")
            logger.info(f"Total warnings: {aggregate_validation['total_warnings']}")
            
            if aggregate_validation['common_issues']:
                logger.info("Most common issues:")
                for issue, count in list(aggregate_validation['common_issues'].most_common(3)):
                    logger.info(f"  {issue}: {count} occurrences")
        
        except Exception as e:
            logger.warning(f"Could not collect additional race data: {e}")
        
        logger.info("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main()