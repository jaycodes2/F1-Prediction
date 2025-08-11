"""
Input forms for race prediction configuration.
"""
import streamlit as st
from datetime import datetime, timedelta
import json


class RaceInputForm:
    """Form components for race prediction input."""
    
    def __init__(self):
        # Import realistic F1 data
        from ..utils.realistic_f1_data import RealisticF1Data
        f1_data = RealisticF1Data()
        self.current_f1_drivers = f1_data.get_driver_list_for_ui()
        
        self.f1_circuits = [
            {'name': 'Bahrain International Circuit', 'location': 'Bahrain'},
            {'name': 'Jeddah Corniche Circuit', 'location': 'Saudi Arabia'},
            {'name': 'Albert Park Circuit', 'location': 'Australia'},
            {'name': 'Baku City Circuit', 'location': 'Azerbaijan'},
            {'name': 'Miami International Autodrome', 'location': 'USA'},
            {'name': 'Autodromo Enzo e Dino Ferrari', 'location': 'Italy (Imola)'},
            {'name': 'Circuit de Monaco', 'location': 'Monaco'},
            {'name': 'Circuit de Barcelona-Catalunya', 'location': 'Spain'},
            {'name': 'Circuit Gilles Villeneuve', 'location': 'Canada'},
            {'name': 'Red Bull Ring', 'location': 'Austria'},
            {'name': 'Silverstone Circuit', 'location': 'United Kingdom'},
            {'name': 'Hungaroring', 'location': 'Hungary'},
            {'name': 'Circuit de Spa-Francorchamps', 'location': 'Belgium'},
            {'name': 'Circuit Zandvoort', 'location': 'Netherlands'},
            {'name': 'Autodromo Nazionale di Monza', 'location': 'Italy'},
            {'name': 'Marina Bay Street Circuit', 'location': 'Singapore'},
            {'name': 'Suzuka International Racing Course', 'location': 'Japan'},
            {'name': 'Losail International Circuit', 'location': 'Qatar'},
            {'name': 'Circuit of the Americas', 'location': 'USA'},
            {'name': 'Autódromo Hermanos Rodríguez', 'location': 'Mexico'},
            {'name': 'Interlagos', 'location': 'Brazil'},
            {'name': 'Las Vegas Strip Circuit', 'location': 'USA'},
            {'name': 'Yas Marina Circuit', 'location': 'UAE'}
        ]
    
    def render_race_input_form(self):
        """Render the main race input form."""
        # Race basic information
        st.markdown("#### 🏁 Race Information")
        
        race_name = st.text_input(
            "Race Name",
            value="Monaco Grand Prix 2024",
            help="Enter the name of the race"
        )
        
        # Circuit selection
        circuit_names = [f"{circuit['name']} ({circuit['location']})" for circuit in self.f1_circuits]
        selected_circuit_idx = st.selectbox(
            "Circuit",
            range(len(circuit_names)),
            format_func=lambda x: circuit_names[x],
            index=4,  # Default to Monaco
            help="Select the race circuit"
        )
        
        circuit = self.f1_circuits[selected_circuit_idx]['name']
        
        # Race date
        race_date = st.date_input(
            "Race Date",
            value=datetime.now() + timedelta(days=7),
            help="Select the race date"
        )
        
        # Session type
        session_type = st.selectbox(
            "Session Type",
            ["race", "qualifying", "sprint"],
            help="Select the session type"
        )
        
        st.markdown("---")
        
        # Driver configuration
        st.markdown("#### 👥 Driver Configuration")
        
        # Number of drivers
        num_drivers = st.slider(
            "Number of Drivers",
            min_value=5,
            max_value=20,
            value=10,
            help="Select number of drivers to include in prediction"
        )
        
        # Driver selection method
        driver_method = st.radio(
            "Driver Selection",
            ["Quick Setup", "Custom Configuration"],
            help="Choose how to configure drivers"
        )
        
        drivers = []
        
        if driver_method == "Quick Setup":
            drivers = self.render_quick_driver_setup(num_drivers)
        else:
            drivers = self.render_custom_driver_setup(num_drivers)
        
        st.markdown("---")
        
        # Weather configuration
        st.markdown("#### 🌤️ Weather Conditions")
        weather = self.render_weather_form()
        
        # Validate and return configuration
        if self.validate_race_config(race_name, circuit, drivers, weather):
            return {
                'race_name': race_name,
                'circuit': circuit,
                'date': datetime.combine(race_date, datetime.min.time()),
                'drivers': drivers,
                'weather': weather,
                'session_type': session_type
            }
        
        return None
    
    def render_quick_driver_setup(self, num_drivers):
        """Render quick driver setup with realistic F1 2024 data."""
        st.info("Using realistic F1 2024 championship data with varied grid positions")
        
        # Import realistic F1 data
        from ..utils.realistic_f1_data import RealisticF1Data
        f1_data = RealisticF1Data()
        
        # Select top drivers by championship points
        selected_drivers = self.current_f1_drivers[:num_drivers]
        
        drivers = []
        import random
        
        # Add some grid position variation (not always championship order)
        grid_positions = list(range(1, num_drivers + 1))
        random.shuffle(grid_positions)
        
        for i, driver in enumerate(selected_drivers):
            # Get realistic driver data
            driver_data = f1_data.get_driver_data(driver['id'])
            team_data = f1_data.get_team_data(driver.get('team', 'Unknown'))
            
            if not driver_data:
                # Fallback for unknown drivers
                driver_data = {
                    'points': driver.get('points', 0),
                    'wins': driver.get('wins', 0),
                    'team': driver.get('team', 'Unknown')
                }
            
            drivers.append({
                'driver_id': driver['id'],
                'name': driver['name'],
                'grid_position': grid_positions[i],
                'championship_points': driver_data.get('points', 0),
                'constructor_points': int(team_data.get('car_rating', 0.6) * 600),
                'wins_season': driver_data.get('wins', 0),
                'constructor_wins': max(0, driver_data.get('wins', 0)),
                'experience_races': min(400, driver_data.get('points', 0) + 100),
                'car_rating': team_data.get('car_rating', 0.6)
            })
        
        # Sort by grid position for display
        drivers.sort(key=lambda x: x['grid_position'])
        
        # Display driver summary with realistic data
        with st.expander("👥 View Realistic Driver Configuration (F1 2024)"):
            for driver in drivers:
                team = next((d['team'] for d in self.current_f1_drivers if d['id'] == driver['driver_id']), 'Unknown')
                st.write(f"P{driver['grid_position']}: {driver['name']} ({team}) - {driver['championship_points']} pts, {driver['wins_season']} wins")
        
        return drivers
    
    def render_custom_driver_setup(self, num_drivers):
        """Render custom driver configuration."""
        st.markdown("Configure each driver individually:")
        
        drivers = []
        
        # Use tabs for better organization
        if num_drivers <= 10:
            # For smaller grids, show all drivers
            for i in range(num_drivers):
                with st.expander(f"Driver {i+1} Configuration"):
                    driver = self.render_single_driver_form(i)
                    if driver:
                        drivers.append(driver)
        else:
            # For larger grids, use columns
            cols = st.columns(2)
            for i in range(num_drivers):
                col = cols[i % 2]
                with col:
                    with st.expander(f"Driver {i+1}"):
                        driver = self.render_single_driver_form(i)
                        if driver:
                            drivers.append(driver)
        
        return drivers
    
    def render_single_driver_form(self, index):
        """Render form for a single driver."""
        # Driver selection
        driver_names = [f"{d['name']} ({d['team']})" for d in self.current_f1_drivers]
        selected_driver_idx = st.selectbox(
            "Driver",
            range(len(driver_names)),
            format_func=lambda x: driver_names[x],
            index=min(index, len(driver_names) - 1),
            key=f"driver_select_{index}"
        )
        
        selected_driver = self.current_f1_drivers[selected_driver_idx]
        
        # Grid position
        grid_position = st.number_input(
            "Grid Position",
            min_value=1,
            max_value=20,
            value=index + 1,
            key=f"grid_pos_{index}"
        )
        
        # Championship points
        championship_points = st.number_input(
            "Championship Points",
            min_value=0,
            max_value=500,
            value=max(0, 400 - index * 20),
            key=f"champ_points_{index}"
        )
        
        # Constructor points
        constructor_points = st.number_input(
            "Constructor Points",
            min_value=0,
            max_value=800,
            value=max(0, 600 - index * 30),
            key=f"const_points_{index}"
        )
        
        return {
            'driver_id': selected_driver['id'],
            'name': selected_driver['name'],
            'grid_position': grid_position,
            'championship_points': championship_points,
            'constructor_points': constructor_points,
            'wins_season': max(0, 10 - index),
            'constructor_wins': max(0, 15 - index),
            'experience_races': 50 + index * 10,
            'car_rating': max(0.5, 1.0 - index * 0.05)
        }
    
    def render_weather_form(self):
        """Render weather conditions form."""
        col1, col2 = st.columns(2)
        
        with col1:
            conditions = st.selectbox(
                "Weather Conditions",
                ["dry", "wet", "mixed"],
                help="Select weather conditions"
            )
            
            track_temp = st.slider(
                "Track Temperature (°C)",
                min_value=10,
                max_value=60,
                value=30,
                help="Track surface temperature"
            )
            
            air_temp = st.slider(
                "Air Temperature (°C)",
                min_value=5,
                max_value=45,
                value=25,
                help="Ambient air temperature"
            )
        
        with col2:
            humidity = st.slider(
                "Humidity (%)",
                min_value=20,
                max_value=100,
                value=60,
                help="Relative humidity percentage"
            )
            
            wind_speed = st.slider(
                "Wind Speed (km/h)",
                min_value=0,
                max_value=50,
                value=10,
                help="Wind speed"
            )
            
            # Grip level based on conditions
            if conditions == "dry":
                grip_level = st.slider("Grip Level", 0.7, 1.0, 0.9)
            elif conditions == "wet":
                grip_level = st.slider("Grip Level", 0.3, 0.7, 0.5)
            else:  # mixed
                grip_level = st.slider("Grip Level", 0.5, 0.8, 0.7)
        
        return {
            'conditions': conditions,
            'track_temp': float(track_temp),
            'air_temp': float(air_temp),
            'humidity': float(humidity),
            'wind_speed': float(wind_speed),
            'grip_level': float(grip_level)
        }
    
    def validate_race_config(self, race_name, circuit, drivers, weather):
        """Validate the race configuration."""
        errors = []
        
        if not race_name.strip():
            errors.append("Race name is required")
        
        if not circuit.strip():
            errors.append("Circuit selection is required")
        
        if not drivers:
            errors.append("At least one driver must be configured")
        
        # Check for duplicate grid positions
        grid_positions = [d['grid_position'] for d in drivers]
        if len(grid_positions) != len(set(grid_positions)):
            errors.append("Duplicate grid positions detected")
        
        # Check for duplicate drivers
        driver_ids = [d['driver_id'] for d in drivers]
        if len(driver_ids) != len(set(driver_ids)):
            errors.append("Duplicate drivers detected")
        
        if not weather:
            errors.append("Weather conditions must be specified")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
            return False
        
        return True
    
    def render_preset_configurations(self):
        """Render preset race configurations."""
        st.markdown("#### 🎯 Preset Configurations")
        
        presets = {
            "Monaco GP 2024": {
                "description": "Monaco street circuit with challenging overtaking",
                "config": {
                    'race_name': 'Monaco Grand Prix 2024',
                    'circuit': 'Circuit de Monaco',
                    'weather': {'conditions': 'dry', 'track_temp': 42.0, 'grip_level': 0.88}
                }
            },
            "Silverstone GP 2024": {
                "description": "British GP with potential wet weather",
                "config": {
                    'race_name': 'British Grand Prix 2024',
                    'circuit': 'Silverstone Circuit',
                    'weather': {'conditions': 'wet', 'track_temp': 18.0, 'grip_level': 0.45}
                }
            },
            "Monza GP 2024": {
                "description": "High-speed Italian GP with slipstream battles",
                "config": {
                    'race_name': 'Italian Grand Prix 2024',
                    'circuit': 'Autodromo Nazionale di Monza',
                    'weather': {'conditions': 'dry', 'track_temp': 35.0, 'grip_level': 0.92}
                }
            }
        }
        
        selected_preset = st.selectbox(
            "Choose Preset",
            list(presets.keys()),
            help="Select a preset race configuration"
        )
        
        if selected_preset:
            preset = presets[selected_preset]
            st.info(f"📝 {preset['description']}")
            
            if st.button(f"Load {selected_preset}"):
                # Store preset in session state
                st.session_state['preset_config'] = preset['config']
                st.success(f"✅ Loaded {selected_preset} configuration")
                st.rerun()
        
        return presets.get(selected_preset, {}).get('config')