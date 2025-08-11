"""
Session state management for the Streamlit application.
"""
import streamlit as st
from datetime import datetime


class SessionStateManager:
    """Manage Streamlit session state."""
    
    def __init__(self):
        self.default_state = {
            'current_page': 'prediction',
            'model_type': 'ensemble',
            'confidence_threshold': 0.7,
            'prediction_history': [],
            'current_prediction': None,
            'user_preferences': {
                'theme': 'default',
                'show_advanced_options': False,
                'default_drivers': 10
            }
        }
    
    def initialize_session_state(self):
        """Initialize session state with default values."""
        for key, value in self.default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def clear_session(self):
        """Clear all session data."""
        for key in list(st.session_state.keys()):
            if key not in ['current_page']:  # Keep navigation state
                del st.session_state[key]
        
        # Reinitialize with defaults
        self.initialize_session_state()
    
    def save_prediction(self, prediction_data):
        """Save a prediction to the session history."""
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []
        
        # Add timestamp if not present
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now()
        
        st.session_state['prediction_history'].append(prediction_data)
        st.session_state['current_prediction'] = prediction_data
    
    def get_prediction_history(self):
        """Get the prediction history."""
        return st.session_state.get('prediction_history', [])
    
    def get_current_prediction(self):
        """Get the current prediction."""
        return st.session_state.get('current_prediction')
    
    def update_user_preferences(self, preferences):
        """Update user preferences."""
        if 'user_preferences' not in st.session_state:
            st.session_state['user_preferences'] = {}
        
        st.session_state['user_preferences'].update(preferences)
    
    def get_user_preferences(self):
        """Get user preferences."""
        return st.session_state.get('user_preferences', self.default_state['user_preferences'])
    
    def set_model_type(self, model_type):
        """Set the selected model type."""
        st.session_state['model_type'] = model_type
    
    def get_model_type(self):
        """Get the selected model type."""
        return st.session_state.get('model_type', 'ensemble')
    
    def set_confidence_threshold(self, threshold):
        """Set the confidence threshold."""
        st.session_state['confidence_threshold'] = threshold
    
    def get_confidence_threshold(self):
        """Get the confidence threshold."""
        return st.session_state.get('confidence_threshold', 0.7)
    
    def get_session_stats(self):
        """Get session statistics."""
        history = self.get_prediction_history()
        
        if not history:
            return {
                'total_predictions': 0,
                'average_confidence': 0,
                'most_predicted_winner': 'None',
                'session_duration': 0
            }
        
        # Calculate statistics
        total_predictions = len(history)
        confidences = [p.get('confidence', 0) for p in history if 'confidence' in p]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Find most predicted winner
        winners = []
        for prediction in history:
            if 'result' in prediction and prediction['result'].predictions:
                winner = prediction['result'].predictions[0]
                winner_name = getattr(winner, 'driver_name', winner.driver_id)
                winners.append(winner_name)
        
        most_predicted_winner = max(set(winners), key=winners.count) if winners else 'None'
        
        # Session duration (simplified)
        if history:
            first_prediction = min(p.get('timestamp', datetime.now()) for p in history)
            session_duration = (datetime.now() - first_prediction).total_seconds() / 60  # minutes
        else:
            session_duration = 0
        
        return {
            'total_predictions': total_predictions,
            'average_confidence': average_confidence,
            'most_predicted_winner': most_predicted_winner,
            'session_duration': session_duration
        }
    
    def export_session_data(self):
        """Export session data for download."""
        session_data = {
            'prediction_history': self.get_prediction_history(),
            'user_preferences': self.get_user_preferences(),
            'session_stats': self.get_session_stats(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        return session_data
    
    def import_session_data(self, session_data):
        """Import session data from uploaded file."""
        try:
            if 'prediction_history' in session_data:
                st.session_state['prediction_history'] = session_data['prediction_history']
            
            if 'user_preferences' in session_data:
                st.session_state['user_preferences'] = session_data['user_preferences']
            
            return True
        except Exception as e:
            st.error(f"Error importing session data: {e}")
            return False