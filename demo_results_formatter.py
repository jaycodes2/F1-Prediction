"""
Demo script for F1 race prediction results formatting and insights.
"""
import numpy as np
from datetime import datetime
from src.services.prediction_engine import PredictionEngine, PredictionRequest
from src.services.results_formatter import ResultsFormatter, InsightGenerator
from src.models.data_models import PositionPrediction

def create_monaco_gp_scenario():
    """Create a realistic Monaco GP scenario with predictions."""
    # Current F1 drivers with realistic data
    drivers_data = [
        {'id': 'VER', 'name': 'Max Verstappen', 'grid': 1, 'points': 393, 'wins': 15},
        {'id': 'LEC', 'name': 'Charles Leclerc', 'grid': 2, 'points': 206, 'wins': 2},
        {'id': 'HAM', 'name': 'Lewis Hamilton', 'grid': 3, 'points': 180, 'wins': 1},
        {'id': 'NOR', 'name': 'Lando Norris', 'grid': 4, 'points': 169, 'wins': 1},
        {'id': 'PIA', 'name': 'Oscar Piastri', 'grid': 5, 'points': 126, 'wins': 0},
        {'id': 'SAI', 'name': 'Carlos Sainz', 'grid': 6, 'points': 200, 'wins': 1},
        {'id': 'RUS', 'name': 'George Russell', 'grid': 7, 'points': 165, 'wins': 1},
        {'id': 'ALO', 'name': 'Fernando Alonso', 'grid': 8, 'points': 62, 'wins': 0},
        {'id': 'STR', 'name': 'Lance Stroll', 'grid': 9, 'points': 12, 'wins': 0},
        {'id': 'GAS', 'name': 'Pierre Gasly', 'grid': 10, 'points': 8, 'wins': 0}
    ]
    
    # Create prediction request
    request = PredictionRequest(
        race_name="Monaco Grand Prix 2024",
        circuit="Circuit de Monaco",
        date=datetime(2024, 5, 26, 15, 0),
        drivers=[
            {
                'driver_id': driver['id'],
                'name': driver['name'],
                'grid_position': driver['grid'],
                'championship_points': driver['points'],
                'wins_season': driver['wins']
            }
            for driver in drivers_data
        ],
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
    
    # Create realistic predictions with some surprises
    predictions = []
    predicted_positions = [1, 3, 2, 4, 6, 5, 7, 9, 8, 10]  # Some position changes
    
    for i, (driver, pred_pos) in enumerate(zip(drivers_data, predicted_positions)):
        # Create probability distribution
        prob_dist = [0.0] * 20
        
        # Main probability at predicted position
        prob_dist[pred_pos - 1] = 0.5
        
        # Spread probability to nearby positions
        if pred_pos > 1:
            prob_dist[pred_pos - 2] = 0.2
        if pred_pos < 20:
            prob_dist[pred_pos] = 0.2
        if pred_pos > 2:
            prob_dist[pred_pos - 3] = 0.05
        if pred_pos < 19:
            prob_dist[pred_pos + 1] = 0.05
        
        # Calculate expected points
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        expected_points = points_system.get(pred_pos, 0)
        
        # Confidence based on grid position vs predicted position
        position_change = abs(driver['grid'] - pred_pos)
        confidence = max(0.5, 0.95 - position_change * 0.1)
        
        pred = PositionPrediction(
            driver_id=driver['id'],
            predicted_position=pred_pos,
            probability_distribution=prob_dist,
            expected_points=float(expected_points),
            confidence_score=confidence
        )
        pred.driver_name = driver['name']
        predictions.append(pred)
    
    # Sort predictions by predicted position
    predictions.sort(key=lambda x: x.predicted_position)
    
    return request, predictions

def create_silverstone_wet_scenario():
    """Create a Silverstone wet weather scenario."""
    drivers_data = [
        {'id': 'HAM', 'name': 'Lewis Hamilton', 'grid': 3, 'wet_skill': 0.95},  # Excellent in wet
        {'id': 'VER', 'name': 'Max Verstappen', 'grid': 1, 'wet_skill': 0.90},
        {'id': 'RUS', 'name': 'George Russell', 'grid': 5, 'wet_skill': 0.85},
        {'id': 'NOR', 'name': 'Lando Norris', 'grid': 2, 'wet_skill': 0.80},
        {'id': 'LEC', 'name': 'Charles Leclerc', 'grid': 4, 'wet_skill': 0.75},
        {'id': 'ALO', 'name': 'Fernando Alonso', 'grid': 8, 'wet_skill': 0.90},  # Veteran wet weather skill
        {'id': 'SAI', 'name': 'Carlos Sainz', 'grid': 6, 'wet_skill': 0.70},
        {'id': 'PIA', 'name': 'Oscar Piastri', 'grid': 7, 'wet_skill': 0.65}  # Rookie in wet
    ]
    
    request = PredictionRequest(
        race_name="British Grand Prix 2024",
        circuit="Silverstone Circuit",
        date=datetime(2024, 7, 7, 15, 0),
        drivers=[
            {
                'driver_id': driver['id'],
                'name': driver['name'],
                'grid_position': driver['grid']
            }
            for driver in drivers_data
        ],
        weather={
            'conditions': 'wet',
            'track_temp': 18.0,
            'air_temp': 15.0,
            'humidity': 95.0,
            'wind_speed': 25.0,
            'grip_level': 0.45
        },
        session_type="race"
    )
    
    # Wet weather predictions favor skilled drivers
    predictions = []
    
    # Reorder based on wet weather skill
    sorted_drivers = sorted(drivers_data, key=lambda x: x['wet_skill'], reverse=True)
    
    for i, driver in enumerate(sorted_drivers):
        pred_pos = i + 1
        
        # Create probability distribution with more uncertainty in wet
        prob_dist = [0.0] * 20
        prob_dist[pred_pos - 1] = 0.3  # Lower certainty in wet
        
        # More spread in wet conditions
        for j in range(max(0, pred_pos - 3), min(20, pred_pos + 3)):
            if j != pred_pos - 1:
                prob_dist[j] = 0.7 / 6  # Spread remaining probability
        
        points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        expected_points = points_system.get(pred_pos, 0)
        
        # Lower confidence in wet conditions
        confidence = 0.6 + driver['wet_skill'] * 0.3
        
        pred = PositionPrediction(
            driver_id=driver['id'],
            predicted_position=pred_pos,
            probability_distribution=prob_dist,
            expected_points=float(expected_points),
            confidence_score=confidence
        )
        pred.driver_name = driver['name']
        predictions.append(pred)
    
    predictions.sort(key=lambda x: x.predicted_position)
    
    return request, predictions

def demo_results_formatting():
    """Demonstrate comprehensive results formatting capabilities."""
    print("🏎️  F1 RACE PREDICTION RESULTS FORMATTING DEMO 🏎️\n")
    
    # Create formatter and insight generator
    formatter = ResultsFormatter()
    insight_generator = InsightGenerator()
    
    print("=== MONACO GRAND PRIX SCENARIO ===\n")
    
    # Create Monaco scenario
    monaco_request, monaco_predictions = create_monaco_gp_scenario()
    
    # Create prediction result
    from src.services.prediction_engine import PredictionResult
    monaco_result = PredictionResult(
        race_name=monaco_request.race_name,
        predictions=monaco_predictions,
        confidence_score=0.82,
        prediction_metadata={'model_type': 'ensemble', 'feature_count': 15},
        generated_at=datetime.now()
    )
    
    print(f"Race: {monaco_request.race_name}")
    print(f"Circuit: {monaco_request.circuit}")
    print(f"Weather: {monaco_request.weather['conditions']} ({monaco_request.weather['track_temp']}°C)")
    print(f"Drivers: {len(monaco_request.drivers)}")
    print()
    
    # Format results
    print("Formatting race results...")
    formatted_monaco = formatter.format_race_result(monaco_result, monaco_request)
    
    print("✓ Results formatted successfully")
    print(f"✓ Generated {len(formatted_monaco.key_insights)} key insights")
    print()
    
    # Display race information
    print("--- RACE INFORMATION ---")
    race_info = formatted_monaco.race_info
    print(f"Overall Confidence: {race_info['overall_confidence']:.3f}")
    print(f"Prediction Model: {race_info['prediction_model']}")
    print(f"Total Drivers: {race_info['total_drivers']}")
    print()
    
    # Display position predictions
    print("--- POSITION PREDICTIONS ---")
    print(f"{'Pos':<4} {'Driver':<20} {'Pred':<5} {'Points':<7} {'Conf':<6} {'Podium%':<8} {'Strengths'}")
    print("-" * 80)
    
    for pred in formatted_monaco.position_predictions[:8]:  # Top 8
        strengths = ', '.join(pred['key_strengths'][:2]) if pred['key_strengths'] else 'None'
        print(f"{pred['position']:<4} {pred['driver_name']:<20} P{pred['predicted_position']:<4} "
              f"{pred['expected_points']:<7.1f} {pred['confidence_score']:<6.3f} "
              f"{pred['podium_probability']:<8.1%} {strengths[:30]}")
    
    print()
    
    # Display podium analysis
    print("--- PODIUM ANALYSIS ---")
    podium = formatted_monaco.podium_analysis
    print("Most Likely Podium:")
    for i, driver in enumerate(podium['most_likely_podium']):
        prob = podium['podium_probabilities'][driver]
        print(f"  {i+1}. {driver} ({prob:.1%} podium probability)")
    
    if podium['podium_battles']:
        print("\nClose Podium Battles:")
        for battle in podium['podium_battles'][:2]:
            drivers = ' vs '.join(battle['drivers'])
            probs = [f"{p:.1%}" for p in battle['probabilities']]
            print(f"  {drivers} ({' vs '.join(probs)})")
    
    if podium['surprise_podium_candidates']:
        print("\nSurprise Podium Candidates:")
        for surprise in podium['surprise_podium_candidates']:
            print(f"  {surprise['driver']} (P{surprise['predicted_position']}, {surprise['podium_probability']:.1%} podium chance)")
    
    print()
    
    # Display overtaking analysis
    print("--- OVERTAKING ANALYSIS ---")
    overtaking = formatted_monaco.overtaking_analysis
    print(f"Track Overtaking Factor: {overtaking['track_overtaking_factor']:.2f}")
    
    if overtaking['most_likely_overtakes']:
        print("\nMost Likely Overtakes:")
        for overtake in overtaking['most_likely_overtakes'][:3]:
            print(f"  {overtake['overtaker_name']} → {overtake['target_name']} "
                  f"({overtake['probability']:.1%} probability)")
    
    if overtaking['overtaking_hotspots']:
        print("\nOvertaking Hotspots:")
        for hotspot in overtaking['overtaking_hotspots']:
            print(f"  {hotspot['position_area']}: {hotspot['overtake_count']} potential overtakes")
    
    print()
    
    # Display key insights
    print("--- KEY INSIGHTS ---")
    for i, insight in enumerate(formatted_monaco.key_insights, 1):
        print(f"{i}. {insight.title}")
        print(f"   Type: {insight.insight_type.title()}")
        print(f"   Confidence: {insight.confidence:.3f}")
        print(f"   {insight.description}")
        if insight.drivers_involved:
            drivers = ', '.join(insight.drivers_involved)
            print(f"   Drivers: {drivers}")
        print()
    
    # Display statistical summary
    print("--- STATISTICAL SUMMARY ---")
    stats = formatted_monaco.statistical_summary
    print(f"Total Expected Points: {stats['points_statistics']['total_expected_points']:.1f}")
    print(f"Points Scoring Drivers: {stats['points_statistics']['points_scoring_drivers']}")
    print(f"Mean Confidence: {stats['confidence_statistics']['mean_confidence']:.3f}")
    print(f"High Confidence Predictions: {stats['confidence_statistics']['high_confidence_predictions']}")
    print(f"Prediction Quality: {stats['prediction_quality']['prediction_certainty'].title()}")
    print()
    
    # Generate advanced insights
    print("--- ADVANCED INSIGHTS ---")
    advanced_insights = insight_generator.generate_advanced_insights(formatted_monaco)
    
    if advanced_insights:
        for i, insight in enumerate(advanced_insights, 1):
            print(f"{i}. {insight.title}")
            print(f"   {insight.description}")
            print(f"   Confidence: {insight.confidence:.3f}")
            print()
    else:
        print("No advanced insights generated for this scenario.")
        print()
    
    # Test JSON serialization
    print("--- SERIALIZATION TEST ---")
    try:
        json_result = formatted_monaco.to_json()
        print(f"✓ JSON serialization successful ({len(json_result)} characters)")
        
        dict_result = formatted_monaco.to_dict()
        print(f"✓ Dictionary serialization successful ({len(dict_result)} keys)")
        print()
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        print()
    
    # Silverstone wet weather scenario
    print("=== SILVERSTONE WET WEATHER SCENARIO ===\n")
    
    silverstone_request, silverstone_predictions = create_silverstone_wet_scenario()
    
    silverstone_result = PredictionResult(
        race_name=silverstone_request.race_name,
        predictions=silverstone_predictions,
        confidence_score=0.65,  # Lower confidence in wet
        prediction_metadata={'model_type': 'ensemble', 'weather_adjusted': True},
        generated_at=datetime.now()
    )
    
    print(f"Race: {silverstone_request.race_name}")
    print(f"Weather: {silverstone_request.weather['conditions']} (Grip: {silverstone_request.weather['grip_level']:.2f})")
    print()
    
    # Format wet weather results
    formatted_silverstone = formatter.format_race_result(silverstone_result, silverstone_request)
    
    print("--- WET WEATHER PREDICTIONS ---")
    print(f"{'Pos':<4} {'Driver':<20} {'Grid':<5} {'Pred':<5} {'Conf':<6} {'Risk Factors'}")
    print("-" * 75)
    
    for pred in formatted_silverstone.position_predictions:
        # Find original grid position
        grid_pos = None
        for driver in silverstone_request.drivers:
            if driver['driver_id'] == pred['driver_id']:
                grid_pos = driver['grid_position']
                break
        
        risks = ', '.join(pred['risk_factors'][:2]) if pred['risk_factors'] else 'None'
        print(f"{pred['position']:<4} {pred['driver_name']:<20} P{grid_pos or 'N/A':<4} "
              f"P{pred['predicted_position']:<4} {pred['confidence_score']:<6.3f} {risks[:25]}")
    
    print()
    
    # Wet weather insights
    print("--- WET WEATHER INSIGHTS ---")
    for insight in formatted_silverstone.key_insights:
        if insight.insight_type in ['weather', 'surprise']:
            print(f"• {insight.title}")
            print(f"  {insight.description}")
            print()
    
    # Comparison between scenarios
    print("=== SCENARIO COMPARISON ===\n")
    
    print(f"{'Metric':<25} {'Monaco (Dry)':<15} {'Silverstone (Wet)':<20}")
    print("-" * 60)
    print(f"{'Overall Confidence':<25} {monaco_result.confidence_score:<15.3f} {silverstone_result.confidence_score:<20.3f}")
    print(f"{'Mean Driver Confidence':<25} {formatted_monaco.statistical_summary['confidence_statistics']['mean_confidence']:<15.3f} "
          f"{formatted_silverstone.statistical_summary['confidence_statistics']['mean_confidence']:<20.3f}")
    print(f"{'High Conf Predictions':<25} {formatted_monaco.statistical_summary['confidence_statistics']['high_confidence_predictions']:<15} "
          f"{formatted_silverstone.statistical_summary['confidence_statistics']['high_confidence_predictions']:<20}")
    print(f"{'Key Insights Generated':<25} {len(formatted_monaco.key_insights):<15} {len(formatted_silverstone.key_insights):<20}")
    print()
    
    # Summary
    print("=" * 60)
    print("🎯 DEMO SUMMARY")
    print("=" * 60)
    
    total_insights = len(formatted_monaco.key_insights) + len(formatted_silverstone.key_insights)
    total_advanced = len(advanced_insights)
    
    print(f"✅ Scenarios analyzed: 2")
    print(f"✅ Total drivers processed: {len(monaco_predictions) + len(silverstone_predictions)}")
    print(f"✅ Key insights generated: {total_insights}")
    print(f"✅ Advanced insights generated: {total_advanced}")
    print(f"✅ Overtaking scenarios analyzed: {len(formatted_monaco.overtaking_analysis['most_likely_overtakes'])}")
    print(f"✅ Podium battles identified: {len(formatted_monaco.podium_analysis['podium_battles'])}")
    
    # Feature highlights
    print("\n🏆 KEY FEATURES DEMONSTRATED:")
    print("• Comprehensive position prediction formatting")
    print("• Podium probability analysis with battle detection")
    print("• Overtaking scenario generation with track factors")
    print("• Multi-type insight generation (winner, surprise, battle, weather)")
    print("• Statistical summaries and confidence analysis")
    print("• Weather impact analysis and driver skill assessment")
    print("• JSON/Dictionary serialization for API integration")
    print("• Advanced insight generation with championship implications")
    
    print("\n🎉 Results formatting demo completed successfully!")

if __name__ == "__main__":
    demo_results_formatting()