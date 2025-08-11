"""
Visualization components for F1 race predictions.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class PredictionVisualizer:
    """Visualization components for prediction results."""
    
    def __init__(self):
        self.f1_colors = {
            'Red Bull Racing': '#0600EF',
            'Ferrari': '#DC143C',
            'Mercedes': '#00D2BE',
            'McLaren': '#FF8700',
            'Aston Martin': '#006F62',
            'Alpine': '#0090FF',
            'Alfa Romeo': '#900000',
            'AlphaTauri': '#2B4562',
            'Haas': '#FFFFFF',
            'Williams': '#005AFF'
        }
    
    def create_position_predictions_chart(self, predictions, title="Position Predictions"):
        """Create a bar chart of position predictions."""
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in predictions[:10]]
        positions = [pred.predicted_position for pred in predictions[:10]]
        confidences = [pred.confidence_score for pred in predictions[:10]]
        
        fig = px.bar(
            x=drivers,
            y=positions,
            color=confidences,
            color_continuous_scale='RdYlGn',
            title=title,
            labels={'x': 'Driver', 'y': 'Predicted Position', 'color': 'Confidence'},
            text=positions
        )
        
        # Reverse y-axis so position 1 is at the top
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            xaxis_tickangle=-45,
            height=500
        )
        
        # Add text on bars
        fig.update_traces(texttemplate='P%{text}', textposition='outside')
        
        return fig
    
    def create_confidence_distribution(self, predictions, title="Confidence Distribution"):
        """Create a histogram of prediction confidences."""
        confidences = [pred.confidence_score for pred in predictions]
        
        fig = px.histogram(
            x=confidences,
            nbins=15,
            title=title,
            labels={'x': 'Confidence Score', 'y': 'Number of Predictions'},
            color_discrete_sequence=['#FF1E00']
        )
        
        # Add vertical lines for confidence thresholds
        fig.add_vline(x=0.8, line_dash="dash", line_color="green", 
                     annotation_text="High Confidence")
        fig.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Confidence")
        
        return fig
    
    def create_podium_probabilities_chart(self, podium_analysis, title="Podium Probabilities"):
        """Create a bar chart of podium probabilities."""
        podium_data = podium_analysis['podium_probabilities']
        
        # Sort by probability and take top 8
        sorted_data = sorted(podium_data.items(), key=lambda x: x[1], reverse=True)[:8]
        drivers = [item[0] for item in sorted_data]
        probabilities = [item[1] for item in sorted_data]
        
        fig = px.bar(
            x=drivers,
            y=probabilities,
            title=title,
            labels={'x': 'Driver', 'y': 'Podium Probability'},
            color=probabilities,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(tickformat='.1%'),
            height=400
        )
        
        return fig
    
    def create_expected_points_chart(self, predictions, title="Expected Championship Points"):
        """Create a chart showing expected championship points."""
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in predictions[:10]]
        points = [pred.expected_points for pred in predictions[:10]]
        
        fig = px.bar(
            x=drivers,
            y=points,
            title=title,
            labels={'x': 'Driver', 'y': 'Expected Points'},
            color=points,
            color_continuous_scale='Blues',
            text=points
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        
        return fig
    
    def create_position_probability_heatmap(self, predictions, title="Position Probability Matrix"):
        """Create a heatmap showing probability of each driver finishing in each position."""
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in predictions[:10]]
        
        # Create probability matrix
        prob_matrix = []
        for pred in predictions[:10]:
            if len(pred.probability_distribution) >= 10:
                prob_matrix.append(pred.probability_distribution[:10])
            else:
                # Pad with zeros if not enough positions
                padded = pred.probability_distribution + [0] * (10 - len(pred.probability_distribution))
                prob_matrix.append(padded[:10])
        
        fig = px.imshow(
            prob_matrix,
            x=[f'P{i+1}' for i in range(10)],
            y=drivers,
            title=title,
            labels={'x': 'Finishing Position', 'y': 'Driver', 'color': 'Probability'},
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_overtaking_analysis_chart(self, overtaking_analysis, title="Overtaking Scenarios"):
        """Create a visualization of overtaking probabilities."""
        overtakes = overtaking_analysis.get('most_likely_overtakes', [])
        
        if not overtakes:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No significant overtaking scenarios predicted",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title=title, height=300)
            return fig
        
        # Extract data for top 5 overtakes
        overtakes = overtakes[:5]
        overtaker_names = [o['overtaker_name'] for o in overtakes]
        target_names = [o['target_name'] for o in overtakes]
        probabilities = [o['probability'] for o in overtakes]
        
        # Create labels for the bars
        labels = [f"{overtaker} → {target}" for overtaker, target in zip(overtaker_names, target_names)]
        
        fig = px.bar(
            x=probabilities,
            y=labels,
            orientation='h',
            title=title,
            labels={'x': 'Overtaking Probability', 'y': 'Overtaking Scenario'},
            color=probabilities,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=max(300, len(overtakes) * 60),
            xaxis=dict(tickformat='.1%')
        )
        
        return fig
    
    def create_race_timeline_simulation(self, predictions, title="Race Position Timeline (Simulated)"):
        """Create a simulated race timeline showing position changes."""
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in predictions[:8]]
        
        # Simulate position changes over race distance
        laps = np.linspace(0, 100, 21)  # 21 points over 100% race distance
        
        fig = go.Figure()
        
        for i, (pred, driver) in enumerate(zip(predictions[:8], drivers)):
            # Start from grid position (simplified - assume grid = index + 1)
            start_pos = i + 1
            end_pos = pred.predicted_position
            
            # Create smooth transition with some randomness
            positions = []
            for lap_pct in laps:
                # Linear interpolation with some noise
                base_pos = start_pos + (end_pos - start_pos) * (lap_pct / 100)
                noise = np.random.normal(0, 0.3) * (1 - lap_pct / 100)  # Less noise towards end
                pos = max(1, min(20, base_pos + noise))
                positions.append(pos)
            
            fig.add_trace(go.Scatter(
                x=laps,
                y=positions,
                mode='lines+markers',
                name=driver,
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Race Progress (%)",
            yaxis_title="Position",
            yaxis=dict(autorange="reversed"),
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_confidence_vs_position_scatter(self, predictions, title="Confidence vs Predicted Position"):
        """Create a scatter plot of confidence vs predicted position."""
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in predictions]
        positions = [pred.predicted_position for pred in predictions]
        confidences = [pred.confidence_score for pred in predictions]
        points = [pred.expected_points for pred in predictions]
        
        fig = px.scatter(
            x=positions,
            y=confidences,
            size=points,
            hover_name=drivers,
            title=title,
            labels={'x': 'Predicted Position', 'y': 'Confidence Score', 'size': 'Expected Points'},
            color=confidences,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            xaxis=dict(autorange="reversed"),  # Position 1 on the right
            height=500
        )
        
        return fig
    
    def create_team_comparison_chart(self, predictions, title="Team Performance Comparison"):
        """Create a chart comparing team performance."""
        # Group predictions by team (simplified - using driver names)
        team_data = {}
        
        for pred in predictions:
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            
            # Simplified team assignment based on driver names
            if 'Verstappen' in driver_name or 'Pérez' in driver_name:
                team = 'Red Bull Racing'
            elif 'Hamilton' in driver_name or 'Russell' in driver_name:
                team = 'Mercedes'
            elif 'Leclerc' in driver_name or 'Sainz' in driver_name:
                team = 'Ferrari'
            elif 'Norris' in driver_name or 'Piastri' in driver_name:
                team = 'McLaren'
            else:
                team = 'Other Teams'
            
            if team not in team_data:
                team_data[team] = {'positions': [], 'points': [], 'confidences': []}
            
            team_data[team]['positions'].append(pred.predicted_position)
            team_data[team]['points'].append(pred.expected_points)
            team_data[team]['confidences'].append(pred.confidence_score)
        
        # Calculate team averages
        teams = []
        avg_positions = []
        total_points = []
        avg_confidences = []
        
        for team, data in team_data.items():
            teams.append(team)
            avg_positions.append(np.mean(data['positions']))
            total_points.append(sum(data['points']))
            avg_confidences.append(np.mean(data['confidences']))
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Average Position', 'Total Expected Points', 'Average Confidence'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average position (lower is better)
        fig.add_trace(
            go.Bar(x=teams, y=avg_positions, name='Avg Position', 
                  marker_color='lightblue'),
            row=1, col=1
        )
        
        # Total points
        fig.add_trace(
            go.Bar(x=teams, y=total_points, name='Total Points', 
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Average confidence
        fig.add_trace(
            go.Bar(x=teams, y=avg_confidences, name='Avg Confidence', 
                  marker_color='lightcoral'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=400
        )
        
        # Reverse y-axis for position (position 1 at top)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        
        return fig
    
    def create_weather_impact_analysis(self, weather_data, predictions, title="Weather Impact Analysis"):
        """Create a visualization showing weather impact on predictions."""
        # Extract weather conditions
        conditions = weather_data.get('conditions', 'dry')
        track_temp = weather_data.get('track_temp', 25)
        humidity = weather_data.get('humidity', 60)
        grip_level = weather_data.get('grip_level', 0.9)
        
        # Create weather summary chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Weather Conditions', 'Track Temperature', 'Humidity Level', 'Grip Level'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Weather conditions indicator
        condition_value = {'dry': 1, 'mixed': 0.5, 'wet': 0}[conditions]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=condition_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Conditions: {conditions.title()}"},
                gauge={'axis': {'range': [None, 1]},
                      'bar': {'color': "darkblue" if conditions == 'wet' else "orange"},
                      'steps': [{'range': [0, 0.33], 'color': "lightgray"},
                               {'range': [0.33, 0.66], 'color': "gray"},
                               {'range': [0.66, 1], 'color': "lightgreen"}]}
            ),
            row=1, col=1
        )
        
        # Track temperature
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=track_temp,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Track Temp (°C)"},
                gauge={'axis': {'range': [0, 60]},
                      'bar': {'color': "red"},
                      'steps': [{'range': [0, 20], 'color': "lightblue"},
                               {'range': [20, 40], 'color': "yellow"},
                               {'range': [40, 60], 'color': "orange"}]}
            ),
            row=1, col=2
        )
        
        # Humidity
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=humidity,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Humidity (%)"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "blue"}}
            ),
            row=2, col=1
        )
        
        # Grip level
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=grip_level,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Grip Level"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "green"},
                      'steps': [{'range': [0, 0.5], 'color': "red"},
                               {'range': [0.5, 0.8], 'color': "yellow"},
                               {'range': [0.8, 1], 'color': "green"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=title,
            height=600
        )
        
        return fig
    
    def create_prediction_summary_dashboard(self, prediction_data):
        """Create a comprehensive dashboard summary."""
        result = prediction_data['result']
        formatted = prediction_data['formatted']
        
        # Create a 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 5 Predictions', 'Confidence Distribution', 
                          'Expected Points', 'Podium Probabilities'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top 5 predictions
        top_5 = result.predictions[:5]
        drivers = [getattr(pred, 'driver_name', pred.driver_id) for pred in top_5]
        positions = [pred.predicted_position for pred in top_5]
        
        fig.add_trace(
            go.Bar(x=drivers, y=positions, name='Position', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Confidence distribution
        confidences = [pred.confidence_score for pred in result.predictions]
        fig.add_trace(
            go.Histogram(x=confidences, name='Confidence', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Expected points
        points = [pred.expected_points for pred in top_5]
        fig.add_trace(
            go.Bar(x=drivers, y=points, name='Points', marker_color='gold'),
            row=2, col=1
        )
        
        # Podium probabilities
        podium_data = formatted.podium_analysis['podium_probabilities']
        top_podium = sorted(podium_data.items(), key=lambda x: x[1], reverse=True)[:5]
        podium_drivers = [item[0] for item in top_podium]
        podium_probs = [item[1] for item in top_podium]
        
        fig.add_trace(
            go.Bar(x=podium_drivers, y=podium_probs, name='Podium Prob', marker_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Race Prediction Dashboard",
            height=800,
            showlegend=False
        )
        
        # Update y-axis for positions (reverse)
        fig.update_yaxes(autorange="reversed", row=1, col=1)
        
        return fig