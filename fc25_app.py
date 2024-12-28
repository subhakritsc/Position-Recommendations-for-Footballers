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

# ------------------------------- FC25 Dataset -----------------------------------------------------------

# fc25_df = pd.read_csv('YOUR PATH OF FC25 DATASET')
fc25_df = pd.read_csv('/Users/subhakritsc/Downloads/Male Players Data.csv')
fc25_columns = [
    'Name', 'Height', 'Weight','Preferred foot', 'Weak foot', 'Skill moves',
    'Acceleration', 'Sprint Speed',
    'Positioning', 'Finishing', 'Shot Power', 'Long Shots', 'Volleys','Penalties',
    'Vision', 'Crossing', 'Free Kick Accuracy','Short Passing', 'Long Passing', 'Curve',
    'Agility','Balance', 'Reactions', 'Ball Control', 'Dribbling', 'Composure',
    'Interceptions', 'Heading Accuracy', 'Def Awareness', 'Standing Tackle', 'Sliding Tackle',
    'Jumping', 'Stamina', 'Strength', 'Aggression',
    'Position',  'Alternative positions', 'Team', 'Nation']

fc25_df = fc25_df.loc[fc25_df['Position'] != 'GK', fc25_columns]
fc25_df.reset_index(drop=True, inplace=True)

fc25_df['Height'] = fc25_df['Height'].str.extract(r'(\d+)')
fc25_df['Weight'] = fc25_df['Weight'].str.extract(r'(\d+)')
fc25_df['Positions'] = fc25_df.apply(lambda row: [row['Position']] + (row['Alternative positions'].split(', ') if pd.notna(row['Alternative positions']) else []), axis=1)
fc25_df['Height'] = pd.to_numeric(fc25_df['Height'], errors='coerce')
fc25_df['Weight'] = pd.to_numeric(fc25_df['Weight'], errors='coerce')

def get_player_input_from_df(df, index):
    row = df.loc[index]
    print(f"Enter the details for player: {row['Name']}")

    # Use the values from the DataFrame row instead of manually inputting
    height_cm = row['Height']
    weight_kg = row['Weight']
    # Calculate height in meters and BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Get player foot and skill details
    preferred_foot = row['Preferred foot']
    weak_foot = row['Weak foot']
    skill_moves = row['Skill moves']

    # Get player skills from the DataFrame
    movement_acceleration = row['Acceleration']
    movement_sprint_speed = row['Sprint Speed']
    mentality_positioning = row['Positioning']
    attacking_finishing = row['Finishing']
    power_shot_power = row['Shot Power']
    power_long_shots = row['Long Shots']
    attacking_volleys = row['Volleys']
    mentality_penalties = row['Penalties']

    mentality_vision = row['Vision']
    attacking_crossing = row['Crossing']
    skill_fk_accuracy = row['Free Kick Accuracy']
    attacking_short_passing = row['Short Passing']
    skill_long_passing = row['Long Passing']
    skill_curve = row['Curve']

    movement_agility = row['Agility']
    movement_balance = row['Balance']
    movement_reactions = row['Reactions']
    skill_ball_control = row['Ball Control']
    skill_dribbling = row['Dribbling']
    mentality_composure = row['Composure']

    mentality_interceptions = row['Interceptions']
    attacking_heading_accuracy = row['Heading Accuracy']
    defending_marking_awareness = row['Def Awareness']
    defending_standing_tackle = row['Standing Tackle']
    defending_sliding_tackle = row['Sliding Tackle']

    power_jumping = row['Jumping']
    power_stamina = row['Stamina']
    power_strength = row['Strength']
    mentality_aggression = row['Aggression']

    # Store all skill values in a list
    skills = [
        attacking_crossing, attacking_finishing, attacking_heading_accuracy, attacking_short_passing,
        attacking_volleys, skill_dribbling, skill_curve, skill_fk_accuracy, skill_long_passing,
        skill_ball_control, movement_acceleration, movement_sprint_speed, movement_agility,
        movement_reactions, movement_balance, power_shot_power, power_jumping, power_stamina,
        power_strength, power_long_shots, mentality_aggression, mentality_interceptions,
        mentality_positioning, mentality_vision, mentality_penalties, mentality_composure,
        defending_marking_awareness, defending_standing_tackle, defending_sliding_tackle
    ]

    # Find the best skill for normalization
    best_skills = max(skills)

    # Normalize all skills by dividing by the best skill
    normalized_skills = [skill / best_skills for skill in skills]

    # Other interesting features
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

    # Convert 'Preferred foot' to numerical value
    preferred_foot = 0 if preferred_foot.lower() == 'right' else 1

    # Return the list of features as required for selection
    return [height_m, weight_kg, bmi, preferred_foot, weak_foot, skill_moves] + normalized_skills + normalized_averages + normalized_additional_features

# ------------------------------ Create website using Streamlit -------------------------------------------
st.set_page_config(layout="wide")
st.title("Position Recommendations for FC25 Players")
# st.markdown("---")
col1, col_space, col2= st.columns([1, 0.2, 3])  # Spaced columns

with col1:
    name = st.text_input("**Search Player Index by Name**", value="")
    filtered_names = [
        {"Index": idx, "Name": short_name}
        for idx, short_name in fc25_df['Name'].items()
        if name.lower() in short_name.lower()
    ]

    if filtered_names:
        # Convert to DataFrame for better display control
        filtered_df = pd.DataFrame(filtered_names)

        # Display the search results in a scrollable table with a fixed height
        st.dataframe(filtered_df, hide_index=True, width =500, height=200)
    else:
        st.markdown('Player Not Found')
 
    search_team = st.text_input("**Search Player Index by Team or Nation**", value="")
    filtered_teams = [
        {"Index": idx, "Name": row['Name']}
        for idx, row in fc25_df.iterrows()
        if search_team.lower() in row['Team'].lower() or search_team.lower() in row['Nation'].lower()
    ]
    
    if filtered_teams:
        # Convert to DataFrame for better display control
        filtered_df_team = pd.DataFrame(filtered_teams)

        # Display the search results in a scrollable table with a fixed height
        st.dataframe(filtered_df_team, hide_index=True, width =500, height=200)
    else:
        st.markdown('Player Not Found')   
    
with col2:
    index = st.number_input("**Enter Player Index**", min_value=0, max_value=len(fc25_df.index)-1, step=1, value=0)
    index = int(index)

    player_name = fc25_df.loc[index, 'Name']
    actual_positions = fc25_df.loc[index, 'Positions']
    team = fc25_df.loc[index, 'Team']
    nation = fc25_df.loc[index, 'Nation']
    
    player = get_player_input_from_df(fc25_df, index)
    sorted_selections = select_and_sort(clf_forest, player, feature_columns, pos_list)
    fig = visualize_results(sorted_selections)
    
    st.markdown(f"Player Name: **{player_name}**")
    st.markdown(f"Actual positions: **{', '.join(actual_positions)}**  |  Team: **{team}**  |  Nation: **{nation}**")
    st.pyplot(fig)