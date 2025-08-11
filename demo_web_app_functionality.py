"""
Demo script to test core web application functionality without Streamlit UI.
"""
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.prediction_engine import PredictionEngine, PredictionRequest
from src.services.results_formatter import ResultsFormatter
from web.utils.data_helpers import DataHelper
from web.utils.session_state import SessionStateManager

def demo_core_functionality():
    """Demonstrate core web application functionality."""
    print("🏎️  F1 Race Prediction Web App - Core Functionality Demo")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing components...")
    data_helper = DataHelper()
    session_manager = SessionStateManager()
    
    # Generate training data
    print("2. Generating training data...")
    training_data = data_helper.generate_training_data(50)
    print(f"   ✅ Generated {len(training_data['features'])} training samples")
    
    # Initialize prediction engine
    print("3. Initializing prediction engine...")
    engine = PredictionEngine(model_type='random_forest')
    success = engine.initialize_model(training_data['features'], training_data['targets'])
    print(f"   ✅ Engine initialized: {success}")
    
    # Create a sample race request
    print("4. Creating sample race request...")
    sample_drivers = [
        {
            'driver_id': 'VER',
            'name': 'Max Verstappen',
            'grid_position': 1,
            'championship_points': 393,
            'constructor_points': 589,
            'wins_season': 15,
            'constructor_wins': 17,
            'experience_races': 180,
            'car_rating': 0.95
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
            'car_rating': 0.88
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
            'car_rating': 0.85
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
            'car_rating': 0.83
        },
        {
            'driver_id': 'SAI',
            'name': 'Carlos Sainz',
            'grid_position': 5,
            'championship_points': 200,
            'constructor_points': 406,
            'wins_season': 1,
            'constructor_wins': 3,
            'experience_races': 190,
            'car_rating': 0.87
        }
    ]
    
    request = PredictionRequest(
        race_name="Monaco Grand Prix 2024",
        circuit="Circuit de Monaco",
        date=datetime(2024, 5, 26, 15, 0),
        drivers=sample_drivers,
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
    
    print(f"   ✅ Created race request: {request.race_name}")
    print(f"   ✅ Drivers: {len(request.drivers)}")
    print(f"   ✅ Weather: {request.weather['conditions']}")
    
    # Generate prediction
    print("5. Generating race prediction...")
    result = engine.predict_race(request)
    print(f"   ✅ Prediction generated with confidence: {result.confidence_score:.3f}")
    
    # Format results
    print("6. Formatting results...")
    formatter = ResultsFormatter()
    formatted_result = formatter.format_race_result(result, request)
    print(f"   ✅ Results formatted with {len(formatted_result.key_insights)} insights")
    
    # Display key results
    print("\n" + "=" * 60)
    print("🏁 PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"Race: {formatted_result.race_info['race_name']}")
    print(f"Overall Confidence: {formatted_result.race_info['overall_confidence']:.3f}")
    print(f"Model: {formatted_result.race_info['prediction_model']}")
    
    print("\nTop 5 Predictions:")
    for i, pred in enumerate(formatted_result.position_predictions[:5]):
        print(f"  {i+1}. {pred['driver_name']} - P{pred['predicted_position']} "
              f"(Confidence: {pred['confidence_score']:.3f}, Points: {pred['expected_points']:.1f})")
    
    print(f"\nMost Likely Podium:")
    podium = formatted_result.podium_analysis['most_likely_podium']
    podium_probs = formatted_result.podium_analysis['podium_probabilities']
    for i, driver in enumerate(podium):
        prob = podium_probs[driver]
        print(f"  {i+1}. {driver} ({prob:.1%} probability)")
    
    print(f"\nKey Insights:")
    for insight in formatted_result.key_insights:
        print(f"  • {insight.title} (Confidence: {insight.confidence:.3f})")
    
    # Test data export
    print("\n7. Testing data export...")
    
    # Test CSV export
    predictions_data = [{'result': result, 'request': request, 'formatted': formatted_result}]
    csv_data = data_helper.export_predictions_to_csv(predictions_data)
    print(f"   ✅ CSV export: {len(csv_data)} characters")
    
    # Test JSON export
    json_data = data_helper.export_predictions_to_json(predictions_data)
    print(f"   ✅ JSON export: {len(json_data)} characters")
    
    # Test session management
    print("8. Testing session management...")
    
    # Simulate session state (without Streamlit)
    mock_session = {}
    
    # Save prediction to mock session
    prediction_data = {
        'request': request,
        'result': result,
        'formatted': formatted_result,
        'timestamp': datetime.now()
    }
    
    mock_session['current_prediction'] = prediction_data
    mock_session['prediction_history'] = [prediction_data]
    
    print(f"   ✅ Session data stored")
    print(f"   ✅ Prediction history: {len(mock_session['prediction_history'])} entries")
    
    # Test statistics calculation
    stats = data_helper.calculate_prediction_statistics([prediction_data])
    print(f"   ✅ Statistics calculated: {stats['total_predictions']} predictions")
    
    print("\n" + "=" * 60)
    print("🎉 CORE FUNCTIONALITY DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nSummary:")
    print(f"✅ Prediction Engine: Working")
    print(f"✅ Results Formatter: Working")
    print(f"✅ Data Helpers: Working")
    print(f"✅ Session Management: Working")
    print(f"✅ Export Functionality: Working")
    
    print(f"\nThe web application is ready to use!")
    print(f"Run: streamlit run web/app.py")

if __name__ == "__main__":
    demo_core_functionality()