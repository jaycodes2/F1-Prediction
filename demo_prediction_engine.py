"""
Demo script for the F1 race prediction engine.
"""
import numpy as np
from datetime import datetime, timedelta
from src.services.prediction_engine import PredictionEngine, PredictionRequest
from src.models.implementations import ModelFactory

def create_realistic_training_data(n_samples=150):
    """Create realistic F1 training data."""
    np.random.seed(42)
    features = []
    targets = []
    
    # F1 driver names for realistic demo
    driver_names = [
        "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris",
        "Oscar Piastri", "Carlos Sainz", "George Russell", "Fernando Alonso",
        "Lance Stroll", "Pierre Gasly", "Alex Albon", "Yuki Tsunoda",
        "Nico Hulkenberg", "Daniel Ricciardo", "Esteban Ocon", "Kevin Magnussen",
        "Valtteri Bottas", "Zhou Guanyu", "Logan Sargeant", "Nyck de Vries"
    ]
    
    for i in range(n_samples):
        # Simulate different race scenarios
        qualifying_pos = np.random.randint(1, 21)
        
        # Driver skill affects performance
        driver_skill = np.random.uniform(0.3, 1.0)
        car_performance = np.random.uniform(0.4, 1.0)
        
        # Weather conditions
        weather_dry = np.random.choice([0, 1], p=[0.25, 0.75])
        track_temp = np.random.uniform(15, 45)
        
        features.append({
            'qualifying_position': qualifying_pos,
            'driver_championship_points': int(driver_skill * 400),
            'constructor_championship_points': int(car_performance * 600),
            'driver_wins_season': int(driver_skill * 10),
            'constructor_wins_season': int(car_performance * 15),
            'track_temperature': track_temp,
            'air_temperature': track_temp - np.random.uniform(0, 10),
            'humidity': np.random.uniform(30, 90),
            'wind_speed': np.random.uniform(0, 20),
            'weather_dry': weather_dry,
            'track_grip': np.random.uniform(0.7, 1.0) * (1.1 if weather_dry else 0.9),
            'fuel_load': np.random.uniform(50, 110),
            'tire_compound': np.random.randint(1, 4),
            'driver_experience': int(driver_skill * 300),
            'car_performance_rating': car_performance,
            'engine_power': np.random.uniform(800, 1000),
            'aerodynamic_efficiency': np.random.uniform(0.6, 1.0)
        })
        
        # Create realistic finishing position
        base_finish = (
            qualifying_pos * 0.5 +  # Qualifying influence
            (1 - driver_skill) * 8 +  # Driver skill
            (1 - car_performance) * 6 +  # Car performance
            np.random.normal(0, 2) * (1.5 if not weather_dry else 1.0)  # Weather chaos
        )
        
        finish_pos = max(1, min(20, int(round(base_finish))))
        targets.append(finish_pos)
    
    return features, targets

def create_monaco_gp_request():
    """Create a realistic Monaco GP prediction request."""
    # Current F1 grid with realistic data
    drivers = [
        {
            'driver_id': 'VER',
            'name': 'Max Verstappen',
            'grid_position': 1,
            'championship_points': 393,
            'constructor_points': 589,
            'wins_season': 15,
            'constructor_wins': 17,
            'experience_races': 180,
            'car_rating': 0.95,
            'fuel_load': 105.0,
            'tire_compound': 2,
            'engine_power': 980.0,
            'aero_efficiency': 0.92
        },
        {
            'driver_id': 'LEC',
            'name': 'Charles Leclerc',
            'grid_position': 2,
            'championship_points': 206,
            'constructor_points': 406,
            'wins_season': 2,
            'constructor_wins': 3,
            'experience_races': 120,
            'car_rating': 0.88,
            'fuel_load': 103.0,
            'tire_compound': 2,
            'engine_power': 970.0,
            'aero_efficiency': 0.89
        },
        {
            'driver_id': 'HAM',
            'name': 'Lewis Hamilton',
            'grid_position': 3,
            'championship_points': 180,
            'constructor_points': 345,
            'wins_season': 1,
            'constructor_wins': 2,
            'experience_races': 320,
            'car_rating': 0.85,
            'fuel_load': 108.0,
            'tire_compound': 1,
            'engine_power': 965.0,
            'aero_efficiency': 0.87
        },
        {
            'driver_id': 'NOR',
            'name': 'Lando Norris',
            'grid_position': 4,
            'championship_points': 169,
            'constructor_points': 295,
            'wins_season': 1,
            'constructor_wins': 1,
            'experience_races': 100,
            'car_rating': 0.83,
            'fuel_load': 106.0,
            'tire_compound': 2,
            'engine_power': 955.0,
            'aero_efficiency': 0.85
        },
        {
            'driver_id': 'PIA',
            'name': 'Oscar Piastri',
            'grid_position': 5,
            'championship_points': 126,
            'constructor_points': 295,
            'wins_season': 0,
            'constructor_wins': 1,
            'experience_races': 40,
            'car_rating': 0.81,
            'fuel_load': 107.0,
            'tire_compound': 2,
            'engine_power': 955.0,
            'aero_efficiency': 0.84
        },
        {
            'driver_id': 'SAI',
            'name': 'Carlos Sainz',
            'grid_position': 6,
            'championship_points': 200,
            'constructor_points': 406,
            'wins_season': 1,
            'constructor_wins': 3,
            'experience_races': 190,
            'car_rating': 0.87,
            'fuel_load': 104.0,
            'tire_compound': 1,
            'engine_power': 970.0,
            'aero_efficiency': 0.88
        },
        {
            'driver_id': 'RUS',
            'name': 'George Russell',
            'grid_position': 7,
            'championship_points': 165,
            'constructor_points': 345,
            'wins_season': 1,
            'constructor_wins': 2,
            'experience_races': 80,
            'car_rating': 0.84,
            'fuel_load': 109.0,
            'tire_compound': 1,
            'engine_power': 965.0,
            'aero_efficiency': 0.86
        },
        {
            'driver_id': 'ALO',
            'name': 'Fernando Alonso',
            'grid_position': 8,
            'championship_points': 62,
            'constructor_points': 74,
            'wins_season': 0,
            'constructor_wins': 0,
            'experience_races': 380,
            'car_rating': 0.75,
            'fuel_load': 110.0,
            'tire_compound': 3,
            'engine_power': 940.0,
            'aero_efficiency': 0.78
        },
        {
            'driver_id': 'STR',
            'name': 'Lance Stroll',
            'grid_position': 9,
            'championship_points': 12,
            'constructor_points': 74,
            'wins_season': 0,
            'constructor_wins': 0,
            'experience_races': 140,
            'car_rating': 0.73,
            'fuel_load': 111.0,
            'tire_compound': 3,
            'engine_power': 940.0,
            'aero_efficiency': 0.77
        },
        {
            'driver_id': 'GAS',
            'name': 'Pierre Gasly',
            'grid_position': 10,
            'championship_points': 8,
            'constructor_points': 16,
            'wins_season': 0,
            'constructor_wins': 0,
            'experience_races': 120,
            'car_rating': 0.70,
            'fuel_load': 112.0,
            'tire_compound': 3,
            'engine_power': 930.0,
            'aero_efficiency': 0.75
        }
    ]
    
    return PredictionRequest(
        race_name="Monaco Grand Prix 2024",
        circuit="Circuit de Monaco",
        date=datetime(2024, 5, 26, 15, 0),
        drivers=drivers,
        weather={
            'conditions': 'dry',
            'track_temp': 42.0,
            'air_temp': 28.0,
            'humidity': 68.0,
            'wind_speed': 12.0,
            'grip_level': 0.88
        },
        session_type="race"
    )

def create_silverstone_gp_request():
    """Create a Silverstone GP request with different conditions."""
    # Same drivers but different grid positions and weather
    drivers = [
        {
            'driver_id': 'HAM',
            'name': 'Lewis Hamilton',
            'grid_position': 1,  # Home advantage
            'championship_points': 180,
            'constructor_points': 345,
            'wins_season': 1,
            'constructor_wins': 2,
            'experience_races': 320,
            'car_rating': 0.85,
            'fuel_load': 105.0,
            'tire_compound': 1,
            'engine_power': 965.0,
            'aero_efficiency': 0.87
        },
        {
            'driver_id': 'VER',
            'name': 'Max Verstappen',
            'grid_position': 2,
            'championship_points': 393,
            'constructor_points': 589,
            'wins_season': 15,
            'constructor_wins': 17,
            'experience_races': 180,
            'car_rating': 0.95,
            'fuel_load': 103.0,
            'tire_compound': 1,
            'engine_power': 980.0,
            'aero_efficiency': 0.92
        },
        {
            'driver_id': 'NOR',
            'name': 'Lando Norris',
            'grid_position': 3,
            'championship_points': 169,
            'constructor_points': 295,
            'wins_season': 1,
            'constructor_wins': 1,
            'experience_races': 100,
            'car_rating': 0.83,
            'fuel_load': 106.0,
            'tire_compound': 2,
            'engine_power': 955.0,
            'aero_efficiency': 0.85
        },
        {
            'driver_id': 'LEC',
            'name': 'Charles Leclerc',
            'grid_position': 4,
            'championship_points': 206,
            'constructor_points': 406,
            'wins_season': 2,
            'constructor_wins': 3,
            'experience_races': 120,
            'car_rating': 0.88,
            'fuel_load': 107.0,
            'tire_compound': 2,
            'engine_power': 970.0,
            'aero_efficiency': 0.89
        },
        {
            'driver_id': 'RUS',
            'name': 'George Russell',
            'grid_position': 5,
            'championship_points': 165,
            'constructor_points': 345,
            'wins_season': 1,
            'constructor_wins': 2,
            'experience_races': 80,
            'car_rating': 0.84,
            'fuel_load': 108.0,
            'tire_compound': 2,
            'engine_power': 965.0,
            'aero_efficiency': 0.86
        }
    ]
    
    return PredictionRequest(
        race_name="British Grand Prix 2024",
        circuit="Silverstone Circuit",
        date=datetime(2024, 7, 7, 15, 0),
        drivers=drivers,
        weather={
            'conditions': 'wet',  # Typical British weather
            'track_temp': 22.0,
            'air_temp': 18.0,
            'humidity': 85.0,
            'wind_speed': 18.0,
            'grip_level': 0.65
        },
        session_type="race"
    )

def demo_prediction_engine():
    """Demonstrate the prediction engine capabilities."""
    print("🏎️  F1 RACE PREDICTION ENGINE DEMO 🏎️\n")
    
    # Create training data
    print("Creating realistic F1 training data...")
    training_features, training_targets = create_realistic_training_data(120)
    print(f"✓ Generated {len(training_features)} training samples")
    print(f"✓ Features per sample: {len(training_features[0])}")
    print()
    
    # Test different model types
    model_types = ['random_forest', 'xgboost', 'ensemble']
    engines = {}
    
    for model_type in model_types:
        print(f"--- Initializing {model_type.upper()} Prediction Engine ---")
        
        try:
            # Create and initialize engine
            engine = PredictionEngine(model_type=model_type, model_config={'random_state': 42})
            success = engine.initialize_model(training_features, training_targets)
            
            if success:
                print(f"✓ {model_type} engine initialized successfully")
                engines[model_type] = engine
                
                # Check model status
                status = engine.get_model_status()
                print(f"✓ Model accuracy: {status['model_accuracies'].get(model_type, 'N/A')}")
            else:
                print(f"✗ Failed to initialize {model_type} engine")
                
        except Exception as e:
            print(f"✗ Error initializing {model_type}: {e}")
        
        print()
    
    if not engines:
        print("❌ No engines initialized successfully")
        return
    
    # Demo single race prediction
    print("=== SINGLE RACE PREDICTION DEMO ===\n")
    
    # Monaco GP prediction
    monaco_request = create_monaco_gp_request()
    print(f"Predicting: {monaco_request.race_name}")
    print(f"Circuit: {monaco_request.circuit}")
    print(f"Drivers: {len(monaco_request.drivers)}")
    print(f"Weather: {monaco_request.weather['conditions']} ({monaco_request.weather['track_temp']}°C)")
    print()
    
    monaco_results = {}
    for model_type, engine in engines.items():
        try:
            result = engine.predict_race(monaco_request)
            monaco_results[model_type] = result
            
            print(f"--- {model_type.upper()} Predictions ---")
            print(f"Overall Confidence: {result.confidence_score:.3f}")
            print("Top 5 Predictions:")
            
            for i, pred in enumerate(result.predictions[:5]):
                driver_name = getattr(pred, 'driver_name', pred.driver_id)
                print(f"  {i+1}. {driver_name} (P{pred.predicted_position}, "
                      f"confidence: {pred.confidence_score:.3f})")
            
            print()
            
        except Exception as e:
            print(f"✗ {model_type} prediction failed: {e}")
            print()
    
    # Demo batch prediction
    print("=== BATCH PREDICTION DEMO ===\n")
    
    # Create multiple race requests
    silverstone_request = create_silverstone_gp_request()
    batch_requests = [monaco_request, silverstone_request]
    
    print(f"Batch predicting {len(batch_requests)} races...")
    
    # Use the best performing engine for batch prediction
    best_engine = engines[list(engines.keys())[0]]  # Use first available
    
    try:
        batch_results = best_engine.predict_batch(batch_requests)
        
        print(f"✓ Batch prediction completed")
        
        for result in batch_results:
            print(f"\n{result.race_name}:")
            print(f"  Confidence: {result.confidence_score:.3f}")
            winner_name = getattr(result.predictions[0], 'driver_name', result.predictions[0].driver_id)
            podium_names = [getattr(p, 'driver_name', p.driver_id) for p in result.predictions[:3]]
            print(f"  Winner prediction: {winner_name}")
            print(f"  Podium: {', '.join(podium_names)}")
        
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
    
    print()
    
    # Demo prediction insights
    print("=== PREDICTION INSIGHTS DEMO ===\n")
    
    if monaco_results:
        # Use first available result
        result = list(monaco_results.values())[0]
        engine = list(engines.values())[0]
        
        insights = engine.get_prediction_insights(result)
        
        print(f"Race: {insights['race_name']}")
        print(f"Total Drivers: {insights['total_drivers']}")
        print(f"Confidence Level: {insights['confidence_level']}")
        print()
        
        print("Most Likely Winner:")
        winner = insights['most_likely_winner']
        print(f"  {winner['driver']} (confidence: {winner['confidence']:.3f})")
        print()
        
        print("Confidence Distribution:")
        conf_dist = insights['confidence_distribution']
        print(f"  High confidence predictions: {conf_dist['high']}")
        print(f"  Medium confidence predictions: {conf_dist['medium']}")
        print(f"  Low confidence predictions: {conf_dist['low']}")
        print()
        
        if insights['biggest_surprises']:
            print("Biggest Surprises:")
            for surprise in insights['biggest_surprises']:
                print(f"  {surprise['driver']} predicted P{surprise['predicted_position']} "
                      f"(surprise factor: {surprise['surprise_factor']:.1f})")
            print()
        
        if insights['closest_battles']:
            print("Closest Battles:")
            for battle in insights['closest_battles']:
                drivers = ' vs '.join(battle['drivers'])
                positions = [f"P{p:.1f}" for p in battle['positions']]
                print(f"  {drivers} ({' vs '.join(positions)})")
            print()
    
    # Demo confidence calculation
    print("=== CONFIDENCE ANALYSIS DEMO ===\n")
    
    if len(monaco_results) > 1:
        # Compare predictions from different models
        predictions_list = []
        model_names = []
        
        for model_type, result in monaco_results.items():
            predictions = [pred.predicted_position for pred in result.predictions]
            predictions_list.append(np.array(predictions))
            model_names.append(model_type)
        
        # Calculate cross-model confidence
        engine = list(engines.values())[0]
        confidence_metrics = engine.confidence_calculator.calculate_prediction_confidence(
            predictions_list,
            [0.85, 0.90, 0.88]  # Mock accuracy scores
        )
        
        print("Cross-Model Confidence Analysis:")
        print(f"  Model Agreement: {confidence_metrics['model_agreement']:.3f}")
        print(f"  Prediction Variance: {confidence_metrics['prediction_variance']:.3f}")
        print(f"  Historical Accuracy: {confidence_metrics['historical_accuracy']:.3f}")
        print(f"  Overall Confidence: {confidence_metrics['overall_confidence']:.3f}")
        print()
        
        # Show prediction comparison
        print("Model Prediction Comparison (Top 3):")
        for i in range(3):
            print(f"  Position {i+1}:")
            for j, model_type in enumerate(model_names):
                result = monaco_results[model_type]
                pred = result.predictions[i]
                driver = getattr(pred, 'driver_name', pred.driver_id)
                pos = pred.predicted_position
                conf = pred.confidence_score
                print(f"    {model_type}: {driver} (P{pos}, conf: {conf:.3f})")
            print()
    
    # Demo caching
    print("=== CACHING DEMO ===\n")
    
    engine = list(engines.values())[0]
    
    print("Testing prediction caching...")
    
    # First prediction (should create cache)
    import time
    start_time = time.time()
    result1 = engine.predict_race(monaco_request)
    first_time = time.time() - start_time
    
    # Second prediction (should use cache)
    start_time = time.time()
    result2 = engine.predict_race(monaco_request)
    second_time = time.time() - start_time
    
    print(f"✓ First prediction time: {first_time:.3f}s")
    print(f"✓ Second prediction time: {second_time:.3f}s")
    if second_time > 0:
        print(f"✓ Cache speedup: {first_time/second_time:.1f}x")
    else:
        print(f"✓ Cache speedup: Very fast (cached)")
    print(f"✓ Cache size: {len(engine.prediction_cache)} entries")
    print()
    
    # Demo accuracy tracking
    print("=== ACCURACY TRACKING DEMO ===\n")
    
    print("Simulating race results and updating model accuracy...")
    
    # Simulate actual race results
    actual_results = [1, 3, 2, 5, 4, 6, 8, 7, 10, 9]  # Monaco results
    predicted_results = [pred.predicted_position for pred in result1.predictions[:10]]
    
    print(f"Actual results:    {actual_results}")
    print(f"Predicted results: {predicted_results}")
    
    # Update accuracy
    engine.update_model_accuracy(actual_results, predicted_results)
    
    # Show updated status
    status = engine.get_model_status()
    print(f"✓ Updated model accuracy: {status['model_accuracies'].get(engine.model_type, 'N/A'):.3f}")
    print()
    
    # Summary
    print("=" * 60)
    print("🎯 DEMO SUMMARY")
    print("=" * 60)
    
    successful_engines = len(engines)
    total_predictions = sum(len(results.predictions) for results in monaco_results.values())
    
    print(f"✅ Engines initialized: {successful_engines}")
    print(f"✅ Total predictions generated: {total_predictions}")
    print(f"✅ Batch predictions: {len(batch_requests)} races")
    print(f"✅ Cache entries: {len(engine.prediction_cache)}")
    
    if monaco_results:
        best_confidence = max(result.confidence_score for result in monaco_results.values())
        print(f"🏆 Best prediction confidence: {best_confidence:.3f}")
        
        winner_predictions = [result.predictions[0].driver_name for result in monaco_results.values()]
        most_common_winner = max(set(winner_predictions), key=winner_predictions.count)
        print(f"🥇 Most predicted winner: {most_common_winner}")
    
    print("\n🎉 Prediction Engine demo completed successfully!")

if __name__ == "__main__":
    demo_prediction_engine()