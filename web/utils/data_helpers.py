"""
Data helper utilities for the web application.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
from .realistic_f1_data import RealisticF1Data


class DataHelper:
    """Helper class for data operations in the web application."""
    
    def __init__(self):
        # Use realistic F1 data
        self.f1_data = RealisticF1Data()
        self.current_f1_drivers = self.f1_data.get_driver_list_for_ui()
    
    def generate_training_data(self, n_samples=100):
        """Generate realistic training data using F1 2024 season data."""
        return self.f1_data.generate_realistic_training_data(n_samples)
    
    def generate_sample_csv(self):
        """Generate a sample CSV file for batch processing."""
        sample_races = [
            {
                'race_name': 'Monaco Grand Prix 2024',
                'circuit': 'Circuit de Monaco',
                'date': '2024-05-26',
                'drivers': json.dumps([
                    {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 1},
                    {'driver_id': 'LEC', 'name': 'Charles Leclerc', 'grid_position': 2},
                    {'driver_id': 'HAM', 'name': 'Lewis Hamilton', 'grid_position': 3}
                ]),
                'weather': json.dumps({
                    'conditions': 'dry',
                    'track_temp': 42.0,
                    'air_temp': 28.0,
                    'humidity': 68.0
                })
            },
            {
                'race_name': 'British Grand Prix 2024',
                'circuit': 'Silverstone Circuit',
                'date': '2024-07-07',
                'drivers': json.dumps([
                    {'driver_id': 'HAM', 'name': 'Lewis Hamilton', 'grid_position': 1},
                    {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 2},
                    {'driver_id': 'NOR', 'name': 'Lando Norris', 'grid_position': 3}
                ]),
                'weather': json.dumps({
                    'conditions': 'wet',
                    'track_temp': 18.0,
                    'air_temp': 15.0,
                    'humidity': 95.0
                })
            },
            {
                'race_name': 'Italian Grand Prix 2024',
                'circuit': 'Autodromo Nazionale di Monza',
                'date': '2024-09-01',
                'drivers': json.dumps([
                    {'driver_id': 'LEC', 'name': 'Charles Leclerc', 'grid_position': 1},
                    {'driver_id': 'VER', 'name': 'Max Verstappen', 'grid_position': 2},
                    {'driver_id': 'SAI', 'name': 'Carlos Sainz', 'grid_position': 3}
                ]),
                'weather': json.dumps({
                    'conditions': 'dry',
                    'track_temp': 35.0,
                    'air_temp': 30.0,
                    'humidity': 55.0
                })
            }
        ]
        
        df = pd.DataFrame(sample_races)
        return df.to_csv(index=False)
    
    def validate_csv_format(self, df):
        """Validate uploaded CSV format."""
        required_columns = ['race_name', 'circuit', 'date', 'drivers', 'weather']
        errors = []
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Validate data types and formats
        for index, row in df.iterrows():
            row_errors = []
            
            # Validate race_name
            if pd.isna(row.get('race_name')) or not str(row.get('race_name')).strip():
                row_errors.append("race_name is required")
            
            # Validate circuit
            if pd.isna(row.get('circuit')) or not str(row.get('circuit')).strip():
                row_errors.append("circuit is required")
            
            # Validate date
            try:
                pd.to_datetime(row.get('date'))
            except:
                row_errors.append("date must be in valid date format")
            
            # Validate drivers JSON
            try:
                drivers = json.loads(row.get('drivers', '[]'))
                if not isinstance(drivers, list) or len(drivers) == 0:
                    row_errors.append("drivers must be a non-empty JSON array")
            except:
                row_errors.append("drivers must be valid JSON")
            
            # Validate weather JSON
            try:
                weather = json.loads(row.get('weather', '{}'))
                if not isinstance(weather, dict):
                    row_errors.append("weather must be a JSON object")
            except:
                row_errors.append("weather must be valid JSON")
            
            if row_errors:
                errors.append(f"Row {index + 1}: {'; '.join(row_errors)}")
        
        return errors
    
    def process_uploaded_csv(self, uploaded_file):
        """Process uploaded CSV file and return validated data."""
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate format
            errors = self.validate_csv_format(df)
            if errors:
                return None, errors
            
            # Process and normalize data
            processed_races = []
            for _, row in df.iterrows():
                try:
                    drivers = json.loads(row['drivers'])
                    weather = json.loads(row['weather'])
                    
                    race_data = {
                        'race_name': str(row['race_name']).strip(),
                        'circuit': str(row['circuit']).strip(),
                        'date': pd.to_datetime(row['date']),
                        'drivers': drivers,
                        'weather': weather
                    }
                    
                    processed_races.append(race_data)
                    
                except Exception as e:
                    errors.append(f"Error processing row: {e}")
            
            if errors:
                return None, errors
            
            return processed_races, []
            
        except Exception as e:
            return None, [f"Error reading CSV file: {e}"]
    
    def export_predictions_to_csv(self, predictions_data):
        """Export predictions to CSV format."""
        rows = []
        
        for prediction in predictions_data:
            if 'result' not in prediction:
                continue
            
            result = prediction['result']
            request = prediction.get('request')
            
            base_row = {
                'race_name': request.race_name if request else 'Unknown',
                'circuit': request.circuit if request else 'Unknown',
                'date': request.date.isoformat() if request else '',
                'overall_confidence': result.confidence_score,
                'prediction_timestamp': prediction.get('timestamp', datetime.now()).isoformat()
            }
            
            # Add driver predictions
            for i, pred in enumerate(result.predictions):
                driver_name = getattr(pred, 'driver_name', pred.driver_id)
                
                row = base_row.copy()
                row.update({
                    'position': i + 1,
                    'driver_id': pred.driver_id,
                    'driver_name': driver_name,
                    'predicted_position': pred.predicted_position,
                    'expected_points': pred.expected_points,
                    'confidence_score': pred.confidence_score
                })
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    
    def export_predictions_to_json(self, predictions_data):
        """Export predictions to JSON format."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions_data),
            'predictions': []
        }
        
        for prediction in predictions_data:
            if 'formatted' in prediction:
                # Use formatted result if available
                formatted_data = prediction['formatted'].to_dict()
                export_data['predictions'].append(formatted_data)
            elif 'result' in prediction:
                # Fallback to basic result
                result = prediction['result']
                basic_data = {
                    'race_name': result.race_name,
                    'confidence_score': result.confidence_score,
                    'predictions': [
                        {
                            'driver_id': pred.driver_id,
                            'predicted_position': pred.predicted_position,
                            'expected_points': pred.expected_points,
                            'confidence_score': pred.confidence_score
                        }
                        for pred in result.predictions
                    ],
                    'generated_at': result.generated_at.isoformat()
                }
                export_data['predictions'].append(basic_data)
        
        return json.dumps(export_data, indent=2)
    
    def create_comparison_dataframe(self, predictions_list):
        """Create a DataFrame for comparing multiple predictions."""
        comparison_data = []
        
        for i, prediction in enumerate(predictions_list):
            if 'result' not in prediction:
                continue
            
            result = prediction['result']
            request = prediction.get('request')
            
            # Winner information
            winner = result.predictions[0]
            winner_name = getattr(winner, 'driver_name', winner.driver_id)
            
            comparison_data.append({
                'Prediction #': i + 1,
                'Race Name': request.race_name if request else f'Prediction {i+1}',
                'Circuit': request.circuit if request else 'Unknown',
                'Overall Confidence': result.confidence_score,
                'Predicted Winner': winner_name,
                'Winner Confidence': winner.confidence_score,
                'Winner Expected Points': winner.expected_points,
                'Total Drivers': len(result.predictions),
                'Timestamp': prediction.get('timestamp', datetime.now())
            })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_prediction_statistics(self, predictions_list):
        """Calculate statistics across multiple predictions."""
        if not predictions_list:
            return {}
        
        # Extract metrics
        confidences = []
        winner_confidences = []
        total_drivers = []
        
        for prediction in predictions_list:
            if 'result' not in prediction:
                continue
            
            result = prediction['result']
            confidences.append(result.confidence_score)
            
            if result.predictions:
                winner_confidences.append(result.predictions[0].confidence_score)
                total_drivers.append(len(result.predictions))
        
        if not confidences:
            return {}
        
        stats = {
            'total_predictions': len(predictions_list),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'average_winner_confidence': np.mean(winner_confidences) if winner_confidences else 0,
            'average_drivers_per_race': np.mean(total_drivers) if total_drivers else 0
        }
        
        return stats
    
    def format_duration(self, seconds):
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def format_percentage(self, value, decimals=1):
        """Format a decimal value as percentage."""
        return f"{value * 100:.{decimals}f}%"
    
    def format_confidence_level(self, confidence):
        """Format confidence score to descriptive level."""
        if confidence > 0.8:
            return "High"
        elif confidence > 0.6:
            return "Medium"
        else:
            return "Low"