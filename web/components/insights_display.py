"""
Components for displaying race insights and analysis.
"""
import streamlit as st
import pandas as pd
from datetime import datetime


class InsightsDisplay:
    """Display components for race insights and analysis."""
    
    def __init__(self):
        self.insight_icons = {
            'winner': '🏆',
            'surprise': '⚡',
            'battle': '⚔️',
            'weather': '🌤️',
            'strategy': '🎯',
            'championship': '👑',
            'team_battle': '🤝',
            'rookie_performance': '🌟'
        }
    
    def display_key_insights(self, insights, title="🔍 Key Race Insights"):
        """Display key insights in an organized format."""
        st.markdown(f"### {title}")
        
        if not insights:
            st.info("No specific insights generated for this prediction.")
            return
        
        # Group insights by type
        insight_groups = {}
        for insight in insights:
            insight_type = insight.insight_type
            if insight_type not in insight_groups:
                insight_groups[insight_type] = []
            insight_groups[insight_type].append(insight)
        
        # Display insights by type
        for insight_type, type_insights in insight_groups.items():
            icon = self.insight_icons.get(insight_type, '💡')
            st.markdown(f"#### {icon} {insight_type.replace('_', ' ').title()}")
            
            for insight in type_insights:
                self.display_single_insight(insight)
            
            st.markdown("---")
    
    def display_single_insight(self, insight):
        """Display a single insight with details."""
        # Confidence color coding
        if insight.confidence > 0.8:
            confidence_color = "🟢"
        elif insight.confidence > 0.6:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        with st.expander(f"{insight.title} {confidence_color}"):
            st.write(insight.description)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Confidence", f"{insight.confidence:.3f}")
            
            with col2:
                if insight.drivers_involved:
                    st.write(f"**Drivers:** {', '.join(insight.drivers_involved)}")
            
            # Supporting data
            if insight.supporting_data:
                st.markdown("**Supporting Data:**")
                self.display_supporting_data(insight.supporting_data)
    
    def display_supporting_data(self, data):
        """Display supporting data in a formatted way."""
        if isinstance(data, dict):
            # Create a formatted display for dictionary data
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if 'probability' in key.lower() or 'confidence' in key.lower():
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value:.1%}")
                    elif 'position' in key.lower():
                        st.write(f"• **{key.replace('_', ' ').title()}:** P{value}")
                    else:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
        else:
            st.write(data)
    
    def display_podium_analysis(self, podium_analysis, title="🏆 Podium Analysis"):
        """Display detailed podium analysis."""
        st.markdown(f"### {title}")
        
        # Most likely podium
        st.markdown("#### Most Likely Podium")
        podium = podium_analysis['most_likely_podium']
        podium_probs = podium_analysis['podium_probabilities']
        
        for i, driver in enumerate(podium):
            prob = podium_probs[driver]
            medal = ['🥇', '🥈', '🥉'][i]
            st.write(f"{medal} **{driver}** - {prob:.1%} probability")
        
        # Podium battles
        if podium_analysis.get('podium_battles'):
            st.markdown("#### Close Podium Battles")
            for battle in podium_analysis['podium_battles']:
                drivers = ' vs '.join(battle['drivers'])
                probs = [f"{p:.1%}" for p in battle['probabilities']]
                st.write(f"⚔️ **{drivers}** ({' vs '.join(probs)})")
        
        # Surprise candidates
        if podium_analysis.get('surprise_podium_candidates'):
            st.markdown("#### Surprise Podium Candidates")
            for surprise in podium_analysis['surprise_podium_candidates']:
                st.write(f"⚡ **{surprise['driver']}** - P{surprise['predicted_position']} "
                        f"with {surprise['podium_probability']:.1%} podium chance")
    
    def display_overtaking_analysis(self, overtaking_analysis, title="🏁 Overtaking Analysis"):
        """Display overtaking scenario analysis."""
        st.markdown(f"### {title}")
        
        # Track overtaking factor
        track_factor = overtaking_analysis['track_overtaking_factor']
        if track_factor > 0.7:
            difficulty = "Easy"
            color = "🟢"
        elif track_factor > 0.4:
            difficulty = "Moderate"
            color = "🟡"
        else:
            difficulty = "Difficult"
            color = "🔴"
        
        st.write(f"**Track Overtaking Difficulty:** {color} {difficulty} (Factor: {track_factor:.2f})")
        
        # Most likely overtakes
        overtakes = overtaking_analysis.get('most_likely_overtakes', [])
        if overtakes:
            st.markdown("#### Most Likely Overtaking Scenarios")
            
            for i, overtake in enumerate(overtakes[:5], 1):
                prob = overtake['probability']
                overtaker = overtake['overtaker_name']
                target = overtake['target_name']
                
                # Probability color coding
                if prob > 0.7:
                    prob_color = "🟢"
                elif prob > 0.4:
                    prob_color = "🟡"
                else:
                    prob_color = "🔴"
                
                st.write(f"{i}. {prob_color} **{overtaker}** → **{target}** ({prob:.1%})")
                
                # Show factors if available
                if overtake.get('factors'):
                    factors_text = ', '.join(overtake['factors'][:2])  # Show first 2 factors
                    st.caption(f"   Factors: {factors_text}")
        
        # Overtaking hotspots
        hotspots = overtaking_analysis.get('overtaking_hotspots', [])
        if hotspots:
            st.markdown("#### Overtaking Hotspots")
            for hotspot in hotspots:
                st.write(f"🔥 **{hotspot['position_area']}**: {hotspot['overtake_count']} potential overtakes")
    
    def display_statistical_summary(self, stats, title="📊 Statistical Summary"):
        """Display statistical summary of predictions."""
        st.markdown(f"### {title}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Position Statistics")
            pos_stats = stats['position_statistics']
            st.metric("Mean Position", f"{pos_stats['mean_position']:.2f}")
            st.metric("Position Spread", f"{pos_stats['position_spread']:.2f}")
            st.metric("Median Position", f"{pos_stats['median_position']:.1f}")
        
        with col2:
            st.markdown("#### Points Statistics")
            points_stats = stats['points_statistics']
            st.metric("Total Expected Points", f"{points_stats['total_expected_points']:.1f}")
            st.metric("Points Scoring Drivers", points_stats['points_scoring_drivers'])
            st.metric("Mean Expected Points", f"{points_stats['mean_expected_points']:.2f}")
        
        with col3:
            st.markdown("#### Confidence Statistics")
            conf_stats = stats['confidence_statistics']
            st.metric("Mean Confidence", f"{conf_stats['mean_confidence']:.3f}")
            st.metric("High Confidence Predictions", conf_stats['high_confidence_predictions'])
            st.metric("Low Confidence Predictions", conf_stats['low_confidence_predictions'])
        
        # Prediction quality indicator
        quality = stats['prediction_quality']['prediction_certainty']
        quality_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
        quality_color = quality_colors.get(quality, '⚪')
        
        st.markdown(f"**Overall Prediction Quality:** {quality_color} {quality.title()}")
    
    def display_confidence_analysis(self, confidence_analysis, title="🎯 Confidence Analysis"):
        """Display detailed confidence analysis."""
        st.markdown(f"### {title}")
        
        distribution = confidence_analysis['distribution']
        
        # Confidence distribution
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟢 High Confidence")
            high_conf = distribution['high_confidence']
            st.metric("Count", high_conf['count'])
            st.metric("Percentage", f"{high_conf['percentage']:.1f}%")
            if high_conf['count'] > 0:
                st.metric("Average", f"{high_conf['average']:.3f}")
        
        with col2:
            st.markdown("#### 🟡 Medium Confidence")
            med_conf = distribution['medium_confidence']
            st.metric("Count", med_conf['count'])
            st.metric("Percentage", f"{med_conf['percentage']:.1f}%")
            if med_conf['count'] > 0:
                st.metric("Average", f"{med_conf['average']:.3f}")
        
        with col3:
            st.markdown("#### 🔴 Low Confidence")
            low_conf = distribution['low_confidence']
            st.metric("Count", low_conf['count'])
            st.metric("Percentage", f"{low_conf['percentage']:.1f}%")
            if low_conf['count'] > 0:
                st.metric("Average", f"{low_conf['average']:.3f}")
        
        # Reliability indicators
        st.markdown("#### Reliability Indicators")
        reliability = confidence_analysis['reliability_indicators']
        
        indicators = [
            ("Consistent Predictions", reliability['consistent_predictions'], 
             "Most predictions have high confidence"),
            ("Uncertain Predictions", reliability['uncertain_predictions'], 
             "Many predictions have low confidence"),
            ("Overall Reliability", reliability['prediction_reliability'], 
             "General prediction reliability level")
        ]
        
        for indicator, value, description in indicators:
            if isinstance(value, bool):
                icon = "✅" if value else "❌"
                st.write(f"{icon} **{indicator}**: {description}")
            else:
                color_map = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
                color = color_map.get(value, '⚪')
                st.write(f"{color} **{indicator}**: {value.title()}")
    
    def display_race_timeline(self, formatted_result, title="⏱️ Race Timeline Insights"):
        """Display race timeline and key moments."""
        st.markdown(f"### {title}")
        
        # Extract key moments from insights
        key_moments = []
        
        for insight in formatted_result.key_insights:
            if insight.insight_type == 'battle':
                key_moments.append({
                    'type': 'Battle',
                    'description': insight.title,
                    'confidence': insight.confidence,
                    'icon': '⚔️'
                })
            elif insight.insight_type == 'surprise':
                key_moments.append({
                    'type': 'Surprise',
                    'description': insight.title,
                    'confidence': insight.confidence,
                    'icon': '⚡'
                })
            elif insight.insight_type == 'strategy':
                key_moments.append({
                    'type': 'Strategy',
                    'description': insight.title,
                    'confidence': insight.confidence,
                    'icon': '🎯'
                })
        
        if key_moments:
            st.markdown("#### Key Race Moments")
            for moment in key_moments:
                confidence_color = "🟢" if moment['confidence'] > 0.8 else "🟡" if moment['confidence'] > 0.6 else "🔴"
                st.write(f"{moment['icon']} **{moment['type']}**: {moment['description']} {confidence_color}")
        else:
            st.info("No specific key moments identified for this race.")
        
        # Race phases analysis
        st.markdown("#### Race Phase Analysis")
        
        phases = [
            ("🏁 Start & Early Laps", "Grid position advantages and early overtaking opportunities"),
            ("🔄 Mid-Race Strategy", "Pit stop windows and strategic battles"),
            ("🏆 Final Push", "Late-race overtaking and position consolidation")
        ]
        
        for phase, description in phases:
            with st.expander(phase):
                st.write(description)
                
                # Add relevant insights for each phase
                relevant_insights = []
                for insight in formatted_result.key_insights:
                    if 'start' in insight.description.lower() or 'early' in insight.description.lower():
                        if phase.startswith("🏁"):
                            relevant_insights.append(insight)
                    elif 'strategy' in insight.description.lower() or 'pit' in insight.description.lower():
                        if phase.startswith("🔄"):
                            relevant_insights.append(insight)
                    elif 'late' in insight.description.lower() or 'final' in insight.description.lower():
                        if phase.startswith("🏆"):
                            relevant_insights.append(insight)
                
                if relevant_insights:
                    for insight in relevant_insights:
                        st.write(f"• {insight.title}")
                else:
                    st.write("No specific insights for this phase.")
    
    def display_driver_spotlight(self, predictions, formatted_result, title="🌟 Driver Spotlight"):
        """Display spotlight on key drivers."""
        st.markdown(f"### {title}")
        
        # Winner spotlight
        winner = predictions[0]
        winner_name = getattr(winner, 'driver_name', winner.driver_id)
        
        st.markdown(f"#### 🏆 Race Winner: {winner_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Position", f"P{winner.predicted_position}")
        
        with col2:
            st.metric("Confidence", f"{winner.confidence_score:.3f}")
        
        with col3:
            st.metric("Expected Points", f"{winner.expected_points:.1f}")
        
        # Find winner in formatted predictions for additional data
        winner_formatted = None
        for pred in formatted_result.position_predictions:
            if pred['driver_id'] == winner.driver_id:
                winner_formatted = pred
                break
        
        if winner_formatted:
            st.write(f"**Podium Probability:** {winner_formatted['podium_probability']:.1%}")
            if winner_formatted['key_strengths']:
                st.write(f"**Key Strengths:** {', '.join(winner_formatted['key_strengths'])}")
        
        # Other notable drivers
        st.markdown("#### 🌟 Other Notable Performances")
        
        # Find drivers with interesting characteristics
        notable_drivers = []
        
        for i, pred in enumerate(predictions):
            driver_name = getattr(pred, 'driver_name', pred.driver_id)
            formatted_pred = formatted_result.position_predictions[i]
            
            # High confidence predictions
            if pred.confidence_score > 0.9 and i > 0:  # Exclude winner
                notable_drivers.append({
                    'name': driver_name,
                    'reason': 'High Confidence',
                    'details': f"P{pred.predicted_position} with {pred.confidence_score:.3f} confidence",
                    'icon': '🎯'
                })
            
            # High podium probability from lower positions
            if formatted_pred['podium_probability'] > 0.3 and pred.predicted_position > 5:
                notable_drivers.append({
                    'name': driver_name,
                    'reason': 'Podium Contender',
                    'details': f"{formatted_pred['podium_probability']:.1%} podium chance from P{pred.predicted_position}",
                    'icon': '🚀'
                })
        
        # Display notable drivers
        for driver in notable_drivers[:3]:  # Show top 3
            st.write(f"{driver['icon']} **{driver['name']}** - {driver['reason']}: {driver['details']}")
    
    def create_insights_summary_card(self, formatted_result):
        """Create a summary card of all insights."""
        insights = formatted_result.key_insights
        
        # Count insights by type
        insight_counts = {}
        for insight in insights:
            insight_type = insight.insight_type
            insight_counts[insight_type] = insight_counts.get(insight_type, 0) + 1
        
        # Overall confidence
        confidences = [insight.confidence for insight in insights]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Create summary
        st.markdown("### 📋 Insights Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Insights", len(insights))
        
        with col2:
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col3:
            high_conf_insights = sum(1 for c in confidences if c > 0.8)
            st.metric("High Confidence", high_conf_insights)
        
        with col4:
            most_common_type = max(insight_counts.items(), key=lambda x: x[1])[0] if insight_counts else "None"
            st.metric("Most Common Type", most_common_type.title())
        
        # Insight type breakdown
        if insight_counts:
            st.markdown("#### Insight Types")
            for insight_type, count in insight_counts.items():
                icon = self.insight_icons.get(insight_type, '💡')
                st.write(f"{icon} **{insight_type.replace('_', ' ').title()}**: {count}")