# ------------------------- Import important tools ---------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# clf_forest = joblib.load('YOUR PATH OF THE MODEL')
clf_forest = joblib.load('/Users/subhakritsc/Downloads/Random Forest Model.pkl')

# --------------------------- Define all lists ---------------------------------------
feature_columns = ['height_m', 'weight_kg', 'bmi', 'preferred_foot', 'weak_foot', 'skill_moves',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing',
        'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
        'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina',
        'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
        'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
        'offensive_avg', 'defensive_avg', 'physical_avg', 'playmaking_avg', 'pace_agility_avg',
        'shooting_avg','offensive_defensive_ratio', 'attacking_contribution', 'defensive_contribution']

pos_list = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM',
            'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM',
            'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

# ------------------------------- Define all functions ------------------------------------------
def select_player_positions(clf, player_features, feature_columns):
    player_df = pd.DataFrame([player_features], columns=feature_columns)

    # Get probability for each position
    position_probabilities = []
    for estimator in clf.estimators_:
        probas = estimator.predict_proba(player_df)[0]
        
        # Check if the classifier was trained on this class
        if len(probas) > 1:
            position_probabilities.append(probas[1])
        else:
            position_probabilities.append(0.0)

    return position_probabilities

def select_and_sort(clf, player_features, feature_columns, pos_list):
    probabilities = select_player_positions(clf, player_features, feature_columns)
    pos_prob_pairs = list(zip(pos_list, probabilities))
    sorted_pairs = sorted(pos_prob_pairs, key=lambda x: x[1], reverse=True)

    return sorted_pairs

def visualize_results(sorted_selections):

    def preprocess(sorted_selections):
        not_found_pos = ['LS', 'RS', 'LF', 'RF', 'LAM', 'RAM', 'LCM', 'RCM', 'LDM', 'RDM', 'LCB', 'RCB']
        filtered_selections = [item for item in sorted_selections if item[0] not in not_found_pos]
        probabilities = dict(filtered_selections)
        return probabilities

    probabilities = preprocess(sorted_selections)

    # Define positions and their coordinates on a football pitch
    positions = {
        'ST': (0.9, 0.5),
        'LW': (0.75, 0.875),
        'RW': (0.75, 0.125),
        'CF': (0.8, 0.5),
        'CAM': (0.7, 0.5),
        'CM': (0.55, 0.5),
        'RM': (0.55, 0.2),
        'LM': (0.55, 0.8),
        'LWB': (0.3, 0.875),
        'RWB': (0.3, 0.125),
        'CDM': (0.4, 0.5),
        'RB': (0.2, 0.25),
        'CB': (0.2, 0.5),
        'LB': (0.2, 0.75)
    }

    # Group positions based on the probability threshold of 0.25
    top_positions = {pos for pos, prob in probabilities.items() if prob >= 0.25}

    # Separate groups for color mapping
    top_probs = [probabilities[pos] for pos in top_positions]
    others_probs = [prob for pos, prob in probabilities.items() if pos not in top_positions]

    # Calculate min and max for each group
    top_min = min(top_probs) if top_probs else 0.25
    top_max = max(top_probs) if top_probs else 1.0
    others_min = min(others_probs) if others_probs else 0.0
    others_max = max(others_probs) if others_probs else 0.24

    # Define a function for interpolating custom colors
    def interpolate_color(color_range, value):
        return tuple((1 - value) * np.array(mcolors.to_rgb(color_range[0])) + value * np.array(mcolors.to_rgb(color_range[1])))

    # Plot the football pitch
    fig, ax = plt.subplots(figsize=(14, 9))

    # Draw the pitch (adjusted for rectangle size)
    pitch = patches.Rectangle([0, 0], 1, 0.7, edgecolor="black", facecolor="lightgreen", lw=2)
    ax.add_patch(pitch)

    # Add halfway line
    plt.plot([0.5, 0.5], [0, 0.7], color="white", linewidth=4)

    # Add penalty areas
    penalty_box_left = patches.Rectangle([0, 0.2], 0.15, 0.3, edgecolor="white", facecolor="none", lw=4)
    penalty_box_right = patches.Rectangle([0.85, 0.2], 0.15, 0.3, edgecolor="white", facecolor="none", lw=4)
    ax.add_patch(penalty_box_left)
    ax.add_patch(penalty_box_right)

    # Add center circle
    center_circle = patches.Circle([0.5, 0.35], 0.1, edgecolor="white", facecolor="none", lw=4)
    ax.add_patch(center_circle)
    plt.scatter(0.5, 0.35, color="white", s=20)  # Center dot

    # Create a ranking for top positions based on their probability
    sorted_top_positions = sorted(top_positions, key=lambda pos: probabilities[pos], reverse=True)
    top_position_ranks = {pos: rank + 1 for rank, pos in enumerate(sorted_top_positions)}

    # Plot player positions with enhanced markers
    for pos, coords in positions.items():
        x, y = coords
        prob = probabilities.get(pos, 0.0)

        # Determine color based on group
        if prob == 0:
            color = "red"
        elif prob >= 0.6:
            color_range = ["greenyellow", "green"]
            norm_prob = (prob - 0.6) / (1.0 - 0.6)
            color = interpolate_color(color_range, norm_prob)
        elif pos in top_positions:
            color_range = ["yellow", "greenyellow"]
            norm_prob = (prob - top_min) / (top_max - top_min) if top_max > top_min else 1
            color = interpolate_color(color_range, norm_prob)
        else:
            color_range = ["red", "orange"]
            norm_prob = (prob - others_min) / (others_max - others_min) if others_max > others_min else 0
            color = interpolate_color(color_range, norm_prob)

        circle = plt.Circle((x, y * 0.7), 0.03, color=color, ec="black", lw=1.5)  # Adjust y for pitch height
        ax.add_artist(circle)
        plt.text(x, y * 0.7 - 0.05, pos, ha='center', fontsize=12, weight="bold", fontfamily="monospace", color="black")

        # Add rank (based on top positions)
        if pos in top_positions:
            rank = top_position_ranks[pos]
            plt.text(x, y * 0.7, f'{rank}', ha='center', va='center', fontsize=12, weight="bold", fontfamily="monospace", color="black")

    # Add border around the pitch after adding other elements
    border = patches.Rectangle([0, 0], 1, 0.7, edgecolor="black", facecolor="none", lw=4, zorder=3)
    ax.add_patch(border)

    # Add title
    # plt.title("Position Recommendations", fontsize=18, fontweight="bold", fontfamily="monospace", color="black")

    # Add the rank list to the right side with a title
    rank_text = "Top Recommended\n\n" + '\n'.join([f"{rank}. {pos} ({probabilities[pos]*100:.1f}%)" for rank, pos in enumerate(sorted_top_positions, start=1)])
    plt.text(1.00, 0.75, rank_text, ha='left', va='top', fontsize=12, weight="bold", fontfamily="monospace", color="black", transform=ax.transAxes)

    plt.axis('off')

    return fig

# ------------------------------ Create website using Streamlit -------------------------------------------
def get_player_input_streamlit():
    # Set page configuration
    st.set_page_config(layout="wide")
    st.title("Position Recommendations for Footballers ⚽")
    st.markdown("Please provide the player's physical details, skill ratings, and additional attributes below:")
    st.markdown("---")
    
    # First Section: Basic details, Pace, Shooting, Passing, Dribbling
    col1, col_space, col2, col_space, col3, col_space, col4 = st.columns([1, 1.4/3, 1, 1.4/3, 1, 1.4/3, 1])  # Spaced columns
    
    with col1:
        st.subheader("Basic Details 📝")
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=1.0, value=175.0)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, step=1.0, value=70.0)
        preferred_foot = st.selectbox("Preferred Foot", ["Right", "Left"])

        # Weak Foot rating
        st.markdown("Weak Foot (1-5)")
        weak_foot = st.feedback("stars", key="weak_foot_rating")
        weak_foot = 0 if weak_foot is None else weak_foot + 1

        st.markdown("Skill Moves (1-5)")
        skill_moves = st.feedback("stars", key="skill_moves_rating")
        skill_moves = 0 if skill_moves is None else skill_moves + 1
        
        # Calculating BMI
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0
        
        st.subheader("Pace 🏃‍♂️")
        movement_acceleration = st.slider("Acceleration", 1, 99, value=50)
        movement_sprint_speed = st.slider("Sprint Speed", 1, 99, value=50)
        
    with col2:
        st.subheader("Shooting 🎯")
        mentality_positioning = st.slider("Positioning", 1, 99, value=50)
        attacking_finishing = st.slider("Finishing", 1, 99, value=50)
        power_shot_power = st.slider("Shot Power", 1, 99, value=50)
        power_long_shots = st.slider("Long Shots", 1, 99, value=50)
        attacking_volleys = st.slider("Volleys", 1, 99, value=50)
        mentality_penalties = st.slider("Penalties", 1, 99, value=50)
        
    with col3:
        st.subheader("Passing 👐")
        mentality_vision = st.slider("Vision", 1, 99, value=50)
        attacking_crossing = st.slider("Crossing", 1, 99, value=50)
        skill_fk_accuracy = st.slider("Free Kick Accuracy", 1, 99, value=50)
        attacking_short_passing = st.slider("Short Passing", 1, 99, value=50)
        skill_long_passing = st.slider("Long Passing", 1, 99, value=50)
        skill_curve = st.slider("Curve", 1, 99, value=50)
    
    with col4:
        st.subheader("Dribbling 💨")
        movement_agility = st.slider("Agility", 1, 99, value=50)
        movement_balance = st.slider("Balance", 1, 99, value=50)
        movement_reactions = st.slider("Reactions", 1, 99, value=50)
        skill_ball_control = st.slider("Ball Control", 1, 99, value=50)
        skill_dribbling = st.slider("Dribbling", 1, 99, value=50)
        mentality_composure = st.slider("Composure", 1, 99, value=50)
    
    # Second Section: Defending, Physical, Position Recommendations
    st.markdown("---") 
    col1, col_space, col2, col_space, col3 = st.columns([1, 0.2, 1, 0.2, 3])  # Spaced columns
    
    with col1:
        st.subheader("Defending 🛡️")
        mentality_interceptions = st.slider("Interceptions", 1, 99, value=50)
        attacking_heading_accuracy = st.slider("Heading Accuracy", 1, 99, value=50)
        defending_marking_awareness = st.slider("Def Awareness", 1, 99, value=50)
        defending_standing_tackle = st.slider("Standing Tackle", 1, 99, value=50)
        defending_sliding_tackle = st.slider("Sliding Tackle", 1, 99, value=50)
        
    with col2:
        st.subheader("Physical 💪")
        power_jumping = st.slider("Jumping", 1, 99, value=50)
        power_stamina = st.slider("Stamina", 1, 99, value=50)
        power_strength = st.slider("Strength", 1, 99, value=50)
        mentality_aggression = st.slider("Aggression", 1, 99, value=50)
    
    with col3:
        st.subheader("Position Recommendations 🔍")
        
        # Normalize skills
        skills = [
            attacking_crossing, attacking_finishing, attacking_heading_accuracy, attacking_short_passing,
            attacking_volleys, skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing,
            skill_ball_control, movement_acceleration, movement_sprint_speed, movement_agility,
            movement_reactions, movement_balance, power_shot_power, power_jumping, power_stamina,
            power_strength, power_long_shots, mentality_aggression, mentality_interceptions,
            mentality_positioning, mentality_vision, mentality_penalties, mentality_composure,
            defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle
        ]
        best_skills = max(skills)
        normalized_skills = [skill / best_skills for skill in skills]

        # Calculate averages for different categories
        offensive_avg = (attacking_crossing + attacking_finishing + attacking_volleys +
                        mentality_positioning + skill_dribbling + skill_curve) / (6 * best_skills)
        defensive_avg = (mentality_aggression + mentality_interceptions + defending_marking_awareness +
                        defending_standing_tackle + defending_sliding_tackle) / (5 * best_skills)
        physical_avg = (power_strength + power_stamina + power_jumping + movement_balance) / (4 * best_skills)
        playmaking_avg = (mentality_vision + attacking_short_passing + skill_long_passing +
                        skill_fk_accuracy + skill_curve) / (5 * best_skills)
        pace_agility_avg = (movement_acceleration + movement_sprint_speed + movement_agility +
                            movement_reactions) / (4 * best_skills)
        shooting_avg = (attacking_finishing + attacking_volleys + power_shot_power +
                        power_long_shots + mentality_penalties) / (5 * best_skills)

        normalized_averages = [offensive_avg, defensive_avg, physical_avg, playmaking_avg, pace_agility_avg, shooting_avg]

        # Additional features
        offensive_defensive_ratio = offensive_avg / (defensive_avg + 1e-5)
        attacking_contribution = (offensive_avg + shooting_avg + playmaking_avg) / 3
        defensive_contribution = (defensive_avg + physical_avg) / 2

        # Normalize the additional features
        normalized_additional_features = [offensive_defensive_ratio, attacking_contribution, defensive_contribution]

        # Return the list of features
        player_features = [height_m, weight_kg, bmi, 0 if preferred_foot == "Right" else 1, weak_foot, skill_moves] + normalized_skills + normalized_averages + normalized_additional_features
        
        # Call the visualization function
        sorted_selections = select_and_sort(clf_forest, player_features, feature_columns, pos_list)
        fig = visualize_results(sorted_selections)
        st.pyplot(fig)

# Call the function
get_player_input_streamlit()