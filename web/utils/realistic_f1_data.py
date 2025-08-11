"""
Realistic F1 2024 data and prediction logic based on actual season performance.
"""
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Tuple


class RealisticF1Data:
    """Provides realistic F1 2024 data and prediction logic."""
    
    def __init__(self):
        # Current 2025 F1 Driver Lineup (Updated)
        self.drivers_2025 = {
            'VER': {
                'name': 'Max Verstappen',
                'team': 'Red Bull Racing',
                'points': 0,  # New season
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 2.1,  # Based on career performance
                'skill_rating': 0.98,
                'consistency': 0.95,
                'wet_weather_skill': 0.96
            },
            'LAW': {
                'name': 'Liam Lawson',
                'team': 'Red Bull Racing',  # Replaced Pérez
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 12.0,  # Estimated based on RB performance
                'skill_rating': 0.78,
                'consistency': 0.75,
                'wet_weather_skill': 0.76
            },
            'HAM': {
                'name': 'Lewis Hamilton',
                'team': 'Ferrari',  # Moved from Mercedes
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 4.5,  # Expected improvement with Ferrari
                'skill_rating': 0.95,
                'consistency': 0.88,
                'wet_weather_skill': 0.98
            },
            'LEC': {
                'name': 'Charles Leclerc',
                'team': 'Ferrari',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 4.8,
                'skill_rating': 0.94,
                'consistency': 0.82,
                'wet_weather_skill': 0.90
            },
            'RUS': {
                'name': 'George Russell',
                'team': 'Mercedes',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 7.2,
                'skill_rating': 0.86,
                'consistency': 0.81,
                'wet_weather_skill': 0.83
            },
            'ANT': {
                'name': 'Andrea Kimi Antonelli',
                'team': 'Mercedes',  # New Mercedes driver
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 10.0,  # Rookie estimate
                'skill_rating': 0.80,
                'consistency': 0.72,
                'wet_weather_skill': 0.75
            },
            'NOR': {
                'name': 'Lando Norris',
                'team': 'McLaren',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 4.2,
                'skill_rating': 0.92,
                'consistency': 0.88,
                'wet_weather_skill': 0.85
            },
            'PIA': {
                'name': 'Oscar Piastri',
                'team': 'McLaren',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 5.6,
                'skill_rating': 0.88,
                'consistency': 0.86,
                'wet_weather_skill': 0.78
            },
            'ALO': {
                'name': 'Fernando Alonso',
                'team': 'Aston Martin',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 10.2,
                'skill_rating': 0.93,
                'consistency': 0.89,
                'wet_weather_skill': 0.94
            },
            'STR': {
                'name': 'Lance Stroll',
                'team': 'Aston Martin',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 12.8,
                'skill_rating': 0.74,
                'consistency': 0.71,
                'wet_weather_skill': 0.69
            },
            'GAS': {
                'name': 'Pierre Gasly',
                'team': 'Alpine',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 12.4,
                'skill_rating': 0.81,
                'consistency': 0.78,
                'wet_weather_skill': 0.77
            },
            'DOO': {
                'name': 'Jack Doohan',
                'team': 'Alpine',  # New Alpine driver
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 14.0,  # Rookie estimate
                'skill_rating': 0.76,
                'consistency': 0.70,
                'wet_weather_skill': 0.72
            },
            'TSU': {
                'name': 'Yuki Tsunoda',
                'team': 'RB',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 12.1,
                'skill_rating': 0.79,
                'consistency': 0.76,
                'wet_weather_skill': 0.74
            },
            'HAD': {
                'name': 'Isack Hadjar',
                'team': 'RB',  # New RB driver
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 15.0,  # Rookie estimate
                'skill_rating': 0.75,
                'consistency': 0.68,
                'wet_weather_skill': 0.70
            },
            'HUL': {
                'name': 'Nico Hülkenberg',
                'team': 'Haas',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 11.8,
                'skill_rating': 0.84,
                'consistency': 0.86,
                'wet_weather_skill': 0.81
            },
            'BEA': {
                'name': 'Oliver Bearman',
                'team': 'Haas',  # Full-time Haas driver
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 13.5,
                'skill_rating': 0.76,
                'consistency': 0.72,
                'wet_weather_skill': 0.70
            },
            'ALB': {
                'name': 'Alex Albon',
                'team': 'Williams',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 16.8,
                'skill_rating': 0.76,
                'consistency': 0.72,
                'wet_weather_skill': 0.71
            },
            'SAI': {
                'name': 'Carlos Sainz',
                'team': 'Williams',  # Moved from Ferrari
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 8.0,  # Expected improvement with Williams
                'skill_rating': 0.89,
                'consistency': 0.84,
                'wet_weather_skill': 0.87
            },
            'BOT': {
                'name': 'Valtteri Bottas',
                'team': 'Kick Sauber',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 16.5,
                'skill_rating': 0.82,
                'consistency': 0.85,
                'wet_weather_skill': 0.80
            },
            'ZHO': {
                'name': 'Zhou Guanyu',
                'team': 'Kick Sauber',
                'points': 0,
                'wins': 0,
                'podiums': 0,
                'poles': 0,
                'fastest_laps': 0,
                'dnfs': 0,
                'avg_finish': 17.9,
                'skill_rating': 0.71,
                'consistency': 0.66,
                'wet_weather_skill': 0.63
            }
        }
        
        # Team performance ratings based on 2024 constructor standings
        self.team_performance = {
            'McLaren': {'car_rating': 0.94, 'reliability': 0.91, 'strategy': 0.88},
            'Ferrari': {'car_rating': 0.91, 'reliability': 0.85, 'strategy': 0.82},
            'Red Bull Racing': {'car_rating': 0.89, 'reliability': 0.93, 'strategy': 0.95},
            'Mercedes': {'car_rating': 0.86, 'reliability': 0.89, 'strategy': 0.90},
            'Aston Martin': {'car_rating': 0.72, 'reliability': 0.78, 'strategy': 0.75},
            'RB': {'car_rating': 0.68, 'reliability': 0.82, 'strategy': 0.73},
            'Haas': {'car_rating': 0.66, 'reliability': 0.76, 'strategy': 0.70},
            'Alpine': {'car_rating': 0.65, 'reliability': 0.74, 'strategy': 0.68},
            'Williams': {'car_rating': 0.58, 'reliability': 0.71, 'strategy': 0.65},
            'Kick Sauber': {'car_rating': 0.52, 'reliability': 0.68, 'strategy': 0.60}
        }
        
        # Circuit characteristics that affect different teams/drivers
        self.circuit_characteristics = {
            'Monaco': {'overtaking': 0.1, 'qualifying_importance': 0.9, 'reliability_factor': 0.8},
            'Silverstone': {'overtaking': 0.7, 'qualifying_importance': 0.6, 'reliability_factor': 0.9},
            'Monza': {'overtaking': 0.8, 'qualifying_importance': 0.5, 'reliability_factor': 0.85},
            'Spa': {'overtaking': 0.75, 'qualifying_importance': 0.55, 'reliability_factor': 0.82},
            'Suzuka': {'overtaking': 0.4, 'qualifying_importance': 0.75, 'reliability_factor': 0.88},
            'Interlagos': {'overtaking': 0.6, 'qualifying_importance': 0.65, 'reliability_factor': 0.75},
            'Singapore': {'overtaking': 0.2, 'qualifying_importance': 0.85, 'reliability_factor': 0.7},
            'Default': {'overtaking': 0.6, 'qualifying_importance': 0.65, 'reliability_factor': 0.85}
        }
    
    def get_driver_data(self, driver_id: str) -> Dict:
        """Get comprehensive driver data."""
        return self.drivers_2024.get(driver_id, {})
    
    def get_team_data(self, team_name: str) -> Dict:
        """Get team performance data."""
        return self.team_performance.get(team_name, {
            'car_rating': 0.60, 'reliability': 0.75, 'strategy': 0.70
        })
    
    def get_circuit_data(self, circuit_name: str) -> Dict:
        """Get circuit characteristics."""
        # Try to match circuit name
        for circuit, data in self.circuit_characteristics.items():
            if circuit.lower() in circuit_name.lower():
                return data
        return self.circuit_characteristics['Default']
    
    def calculate_realistic_race_result(self, drivers: List[Dict], weather: Dict, 
                                      circuit: str = "Default") -> List[Tuple[str, float]]:
        """
        Calculate realistic race results based on multiple factors.
        Returns list of (driver_id, finish_position_probability) tuples.
        """
        circuit_data = self.get_circuit_data(circuit)
        is_wet = weather.get('conditions', 'dry') != 'dry'
        
        driver_scores = []
        
        for driver in drivers:
            driver_id = driver.get('driver_id', '')
            grid_pos = driver.get('grid_position', 20)
            
            driver_data = self.get_driver_data(driver_id)
            if not driver_data:
                # Unknown driver, assign average values
                driver_data = {
                    'skill_rating': 0.75,
                    'consistency': 0.70,
                    'wet_weather_skill': 0.70,
                    'team': 'Unknown'
                }
            
            team_data = self.get_team_data(driver_data.get('team', 'Unknown'))
            
            # Base score from driver skill
            base_score = driver_data['skill_rating'] * 100
            
            # Car performance impact
            car_impact = team_data['car_rating'] * 80
            
            # Qualifying position impact (varies by circuit)
            quali_impact = (21 - grid_pos) * circuit_data['qualifying_importance'] * 3
            
            # Weather impact
            if is_wet:
                weather_impact = driver_data['wet_weather_skill'] * 30
            else:
                weather_impact = 15  # Neutral impact in dry
            
            # Consistency factor (reduces variance)
            consistency_factor = driver_data['consistency']
            
            # Reliability factor
            reliability_impact = team_data['reliability'] * 20
            
            # Random race factors (incidents, strategy, etc.)
            race_randomness = random.gauss(0, 15 * (1 - consistency_factor))
            
            # Strategy impact
            strategy_impact = team_data['strategy'] * 10 + random.gauss(0, 5)
            
            # Calculate total score
            total_score = (base_score + car_impact + quali_impact + 
                          weather_impact + reliability_impact + 
                          race_randomness + strategy_impact)
            
            # Add some circuit-specific randomness
            if circuit_data['overtaking'] > 0.7:  # High overtaking circuits
                total_score += random.gauss(0, 10)
            elif circuit_data['overtaking'] < 0.3:  # Processional circuits
                total_score += random.gauss(0, 5)
            
            driver_scores.append((driver_id, total_score, driver_data['name']))
        
        # Sort by score (higher is better)
        driver_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to position probabilities
        results = []
        for i, (driver_id, score, name) in enumerate(driver_scores):
            # Calculate confidence based on score gap
            if i == 0:
                confidence = min(0.95, 0.7 + (score - driver_scores[1][1]) / 100)
            else:
                confidence = max(0.3, 0.8 - i * 0.05)
            
            results.append((driver_id, i + 1, confidence, name))
        
        return results
    
    def generate_realistic_training_data(self, n_samples: int = 150) -> Dict:
        """Generate realistic training data based on 2024 season patterns."""
        features = []
        targets = []
        
        # Use different random seed each time
        random.seed(int(datetime.now().timestamp()) % 10000)
        np.random.seed(int(datetime.now().timestamp()) % 10000)
        
        circuits = list(self.circuit_characteristics.keys())
        weather_conditions = ['dry', 'wet', 'mixed']
        
        for _ in range(n_samples):
            # Select random driver and circuit
            driver_id = random.choice(list(self.drivers_2024.keys()))
            driver_data = self.drivers_2024[driver_id]
            circuit = random.choice(circuits)
            circuit_data = self.circuit_characteristics[circuit]
            
            # Generate realistic race conditions
            weather_condition = np.random.choice(weather_conditions, p=[0.75, 0.15, 0.10])
            is_wet = weather_condition != 'dry'
            
            # Grid position based on driver's typical qualifying performance
            base_quali = max(1, min(20, int(np.random.normal(driver_data['avg_finish'], 3))))
            grid_position = max(1, min(20, base_quali + random.randint(-2, 2)))
            
            # Weather parameters
            if is_wet:
                track_temp = np.random.uniform(10, 25)
                air_temp = track_temp - np.random.uniform(0, 5)
                humidity = np.random.uniform(80, 100)
                track_grip = np.random.uniform(0.3, 0.7)
            else:
                track_temp = np.random.uniform(25, 55)
                air_temp = track_temp - np.random.uniform(5, 15)
                humidity = np.random.uniform(30, 80)
                track_grip = np.random.uniform(0.8, 1.0)
            
            team_data = self.team_performance[driver_data['team']]
            
            # Create feature vector
            features.append({
                'qualifying_position': grid_position,
                'driver_championship_points': driver_data['points'],
                'constructor_championship_points': int(team_data['car_rating'] * 600),
                'driver_wins_season': driver_data['wins'],
                'constructor_wins_season': max(0, driver_data['wins'] + random.randint(-1, 2)),
                'track_temperature': track_temp,
                'air_temperature': air_temp,
                'humidity': humidity,
                'wind_speed': np.random.uniform(0, 20),
                'weather_dry': 1 if not is_wet else 0,
                'track_grip': track_grip,
                'fuel_load': np.random.uniform(50, 110),
                'tire_compound': random.randint(1, 3),
                'driver_experience': min(400, driver_data['points'] + random.randint(50, 150)),
                'car_performance_rating': team_data['car_rating'],
                'engine_power': np.random.uniform(850, 1000),
                'aerodynamic_efficiency': team_data['car_rating'] * np.random.uniform(0.9, 1.0),
                'driver_skill_rating': driver_data['skill_rating'],
                'consistency_rating': driver_data['consistency'],
                'wet_weather_skill': driver_data['wet_weather_skill'],
                'reliability_rating': team_data['reliability'],
                'strategy_rating': team_data['strategy'],
                'circuit_overtaking_factor': circuit_data['overtaking'],
                'qualifying_importance': circuit_data['qualifying_importance']
            })
            
            # Calculate realistic finishing position
            # Base position from grid
            base_finish = grid_position
            
            # Driver skill adjustment
            skill_adjustment = (0.95 - driver_data['skill_rating']) * 8
            
            # Car performance adjustment
            car_adjustment = (0.95 - team_data['car_rating']) * 6
            
            # Weather adjustment
            if is_wet:
                weather_adjustment = (0.95 - driver_data['wet_weather_skill']) * 4
            else:
                weather_adjustment = random.gauss(0, 1)
            
            # Circuit-specific adjustment
            if circuit_data['overtaking'] > 0.7:
                overtaking_adjustment = random.gauss(0, 3)
            else:
                overtaking_adjustment = random.gauss(0, 1.5)
            
            # Reliability factor (chance of DNF or major issues)
            if random.random() > team_data['reliability']:
                reliability_adjustment = random.randint(5, 15)  # Major issue
            else:
                reliability_adjustment = 0
            
            # Strategy factor
            strategy_adjustment = (0.90 - team_data['strategy']) * 2 + random.gauss(0, 1)
            
            # Calculate final position
            final_position = (base_finish + skill_adjustment + car_adjustment + 
                            weather_adjustment + overtaking_adjustment + 
                            reliability_adjustment + strategy_adjustment)
            
            # Ensure position is within valid range
            final_position = max(1, min(20, int(round(final_position))))
            targets.append(final_position)
        
        return {
            'features': features,
            'targets': targets
        }
    
    def get_driver_list_for_ui(self) -> List[Dict]:
        """Get formatted driver list for UI components."""
        drivers = []
        for driver_id, data in self.drivers_2024.items():
            drivers.append({
                'id': driver_id,
                'name': data['name'],
                'team': data['team'],
                'points': data['points'],
                'wins': data['wins'],
                'skill_rating': data['skill_rating'],
                'consistency': data['consistency']
            })
        
        # Sort by championship points
        drivers.sort(key=lambda x: x['points'], reverse=True)
        return drivers
    
    def predict_race_realistic(self, drivers: List[Dict], weather: Dict, 
                             circuit: str = "Default") -> List[Dict]:
        """
        Generate realistic race predictions with proper probabilities.
        """
        results = self.calculate_realistic_race_result(drivers, weather, circuit)
        
        predictions = []
        for driver_id, position, confidence, name in results:
            # Calculate expected points based on F1 points system
            points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 10
            expected_points = points_system[position - 1] if position <= 10 else 0
            
            # Add some variance to expected points based on confidence
            if position <= 10:
                expected_points *= confidence
            
            predictions.append({
                'driver_id': driver_id,
                'driver_name': name,
                'predicted_position': position,
                'confidence_score': confidence,
                'expected_points': round(expected_points, 1)
            })
        
        return predictions