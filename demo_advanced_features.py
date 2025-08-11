"""
Demo script for advanced feature engineering.
"""
import logging
import json
from src.features.advanced_features import AdvancedFeatureEngineer, TrackSpecificFeatures, WeatherFeatures
from src.data.combined_collector import CombinedDataCollector

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate advanced feature engineering functionality."""
    # Initialize components
    engineer = AdvancedFeatureEngineer()
    collector = CombinedDataCollector()
    track_features = TrackSpecificFeatures()
    weather_features = WeatherFeatures()
    
    try:
        logger.info("=== Advanced Feature Engineering Demo ===")
        
        # Collect historical data for feature engineering
        logger.info("\nCollecting historical race data...")
        historical_races = []
        
        # Collect races from 2023 for feature engineering
        for round_num in range(1, 6):  # First 5 races of 2023
            try:
                race_data = collector.collect_race_data(2023, round_num)
                historical_races.append(race_data)
                logger.info(f"Collected 2023 Round {round_num}: {race_data.race_name}")
            except Exception as e:
                logger.warning(f"Could not collect 2023 Round {round_num}: {e}")
                continue
        
        if len(historical_races) < 3:
            logger.error("Insufficient historical data for advanced feature engineering demo.")
            return
        
        # Use the last race as our target for prediction
        target_race = historical_races[-1]
        historical_data = historical_races[:-1]
        
        logger.info(f"\nTarget Race: {target_race.race_name}")
        logger.info(f"Historical Data: {len(historical_data)} races")
        
        # Demonstrate track-specific features
        logger.info("\n=== Track-Specific Features Demo ===")
        circuit_id = target_race.circuit_id
        logger.info(f"Analyzing track characteristics for: {circuit_id}")
        
        track_characteristics = track_features.get_track_features(circuit_id)
        logger.info("Track Characteristics:")
        for characteristic, value in track_characteristics.items():
            logger.info(f"  {characteristic}: {value}")
        
        # Demonstrate track suitability calculation
        sample_driver_profile = {
            'overtaking_skill': 0.8,
            'wet_weather_skill': 0.7,
            'consistency': 0.9
        }
        
        suitability = track_features.calculate_track_suitability(circuit_id, sample_driver_profile)
        logger.info(f"\nTrack suitability for sample driver profile: {suitability:.3f}")
        
        # Compare with different tracks
        logger.info("\nTrack suitability comparison:")
        test_circuits = ['monaco', 'monza', 'silverstone', 'spa']
        for test_circuit in test_circuits:
            test_suitability = track_features.calculate_track_suitability(test_circuit, sample_driver_profile)
            logger.info(f"  {test_circuit}: {test_suitability:.3f}")
        
        # Demonstrate weather features
        logger.info("\n=== Weather Features Demo ===")
        logger.info(f"Weather conditions for {target_race.race_name}:")
        logger.info(f"  Temperature: {target_race.weather.temperature}°C")
        logger.info(f"  Humidity: {target_race.weather.humidity}%")
        logger.info(f"  Rainfall: {'Yes' if target_race.weather.rainfall else 'No'}")
        logger.info(f"  Wind Speed: {target_race.weather.wind_speed} km/h")
        
        # Extract advanced weather features
        weather_feature_dict = weather_features.extract_weather_features(target_race.weather)
        logger.info("\nAdvanced Weather Features:")
        weather_feature_samples = [
            'temp_optimal', 'weather_extreme', 'humidity_high', 
            'wind_strong', 'pressure_normal'
        ]
        
        for feature in weather_feature_samples:
            if feature in weather_feature_dict:
                logger.info(f"  {feature}: {weather_feature_dict[feature]:.3f}")
        
        # Demonstrate comprehensive feature engineering
        logger.info("\n=== Comprehensive Feature Engineering ===")
        logger.info("Engineering features for all drivers in the target race...")
        
        try:
            engineered_features = engineer.engineer_race_features(target_race, historical_data)
            
            logger.info(f"Successfully engineered features for {len(engineered_features)} drivers")
            
            # Show feature statistics
            if engineered_features:
                sample_driver = list(engineered_features.keys())[0]
                sample_features = engineered_features[sample_driver]
                
                logger.info(f"\nFeature statistics for {sample_driver}:")
                logger.info(f"  Total features: {len(sample_features)}")
                
                # Categorize features
                feature_categories = {
                    'rolling': [f for f in sample_features.keys() if 'rolling' in f],
                    'track': [f for f in sample_features.keys() if 'track' in f],
                    'weather': [f for f in sample_features.keys() if 'weather' in f],
                    'constructor': [f for f in sample_features.keys() if 'constructor' in f],
                    'grid': [f for f in sample_features.keys() if 'grid' in f],
                    'interaction': [f for f in sample_features.keys() if 'interaction' in f],
                    'poly': [f for f in sample_features.keys() if 'poly' in f],
                    'bin': [f for f in sample_features.keys() if 'bin' in f]
                }
                
                logger.info("  Feature categories:")
                for category, features in feature_categories.items():
                    if features:
                        logger.info(f"    {category}: {len(features)} features")
                
                # Show sample features from each category
                logger.info(f"\nSample features for {sample_driver}:")
                
                # Base features
                base_features = ['recent_form', 'constructor_performance', 'track_experience']
                for feature in base_features:
                    if feature in sample_features:
                        logger.info(f"  {feature}: {sample_features[feature]:.3f}")
                
                # Rolling features (show a few)
                rolling_features = [f for f in sample_features.keys() if 'rolling_3' in f][:3]
                for feature in rolling_features:
                    logger.info(f"  {feature}: {sample_features[feature]:.3f}")
                
                # Track features (show a few)
                track_feature_samples = [f for f in sample_features.keys() if 'track_' in f][:3]
                for feature in track_feature_samples:
                    logger.info(f"  {feature}: {sample_features[feature]:.3f}")
                
                # Weather features (show a few)
                weather_feature_samples = [f for f in sample_features.keys() if 'weather_' in f][:3]
                for feature in weather_feature_samples:
                    logger.info(f"  {feature}: {sample_features[feature]:.3f}")
            
            # Validate engineered features
            logger.info("\n=== Feature Validation ===")
            validation_report = engineer.validate_engineered_features(engineered_features)
            
            logger.info("Feature validation report:")
            logger.info(f"  Total drivers: {validation_report['total_drivers']}")
            logger.info(f"  Total features: {validation_report['total_features']}")
            logger.info(f"  Quality score: {validation_report['quality_score']:.3f}")
            
            if validation_report['missing_features']:
                logger.info(f"  Missing features ({len(validation_report['missing_features'])}): "
                          f"{validation_report['missing_features'][:5]}...")  # Show first 5
            else:
                logger.info("  No missing features detected")
            
            logger.info("  Feature counts by category:")
            for category, count in validation_report['feature_counts'].items():
                logger.info(f"    {category}: {count}")
        
        except Exception as e:
            logger.error(f"Failed to engineer comprehensive features: {e}")
        
        # Demonstrate comparative features
        logger.info("\n=== Comparative Features Demo ===")
        try:
            comparative_features = engineer.engineer_comparative_features(target_race, historical_data)
            
            if comparative_features:
                logger.info(f"Generated {len(comparative_features)} comparative features")
                
                # Show field-level features
                field_features = [f for f in comparative_features.keys() if 'field' in f]
                for feature in field_features:
                    logger.info(f"  {feature}: {comparative_features[feature]}")
                
                # Show head-to-head features (first few)
                h2h_features = [f for f in comparative_features.keys() if '_vs_' in f][:5]
                for feature in h2h_features:
                    logger.info(f"  {feature}: {comparative_features[feature]:.3f}")
            else:
                logger.info("No comparative features generated (insufficient data)")
        
        except Exception as e:
            logger.warning(f"Could not generate comparative features: {e}")
        
        # Show feature importance weights
        logger.info("\n=== Feature Importance Weights ===")
        importance_weights = engineer.get_feature_importance_weights()
        
        logger.info("Feature category importance weights:")
        for category, weight in importance_weights.items():
            logger.info(f"  {category}: {weight:.3f} ({weight*100:.1f}%)")
        
        # Demonstrate driver comparison
        if len(engineered_features) >= 2:
            logger.info("\n=== Driver Feature Comparison ===")
            drivers = list(engineered_features.keys())[:2]  # Compare first two drivers
            
            logger.info(f"Comparing {drivers[0]} vs {drivers[1]}:")
            
            # Compare key features
            key_features = ['recent_form', 'rolling_3_avg_position', 'track_specific_avg_position', 'constructor_avg_points']
            
            for feature in key_features:
                if feature in engineered_features[drivers[0]] and feature in engineered_features[drivers[1]]:
                    val1 = engineered_features[drivers[0]][feature]
                    val2 = engineered_features[drivers[1]][feature]
                    
                    better_driver = drivers[0] if val1 > val2 else drivers[1]
                    if 'position' in feature:  # Lower is better for positions
                        better_driver = drivers[0] if val1 < val2 else drivers[1]
                    
                    logger.info(f"  {feature}: {drivers[0]}={val1:.3f}, {drivers[1]}={val2:.3f} "
                              f"(Advantage: {better_driver})")
        
        logger.info("\n=== Advanced feature engineering demo completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    main()