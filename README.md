# Position Recommendations for Footballers Using Python and Scikit-Learn


**[My Project Notebook](https://github.com/subhakritsc/Position-Recommendations-for-Footballers/blob/main/Football%20Position%20Selection%20Test.ipynb)**


## Objective

This project aims to help football players and coaches select the most suitable playing position for a player based on their physical attributes and skills. Using machine learning techniques, the model selects and ranks the best positions for a player, assisting in optimal team selection and player development.


## Dataset

The project uses two datasets for model training and testing:

1. [EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) by [Stefano Leone](https://www.kaggle.com/stefanoleone992): Used for **Training** the model.

2. [EA SPORTS FC 25 DATABASE, RATINGS AND STATS](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats) by [Davis Nyagami](https://www.kaggle.com/nyagami): Used for **Testing** the model and visualization.


## Key Steps in the Project

1. **Data Cleaning and Feature Extraction**:
    - **Data Filtering**: Excluded goalkeepers and selected FIFA versions 22-24.
    - **Position Encoding**: Converted player positions into binary vectors.
    - **Feature Engineering**: Calculated BMI, normalized skills, and created category averages (offensive, defensive, physical, etc.).
    - **Additional Features**: Added attacking/defensive contributions and ratio.

2. **Selecting and Training the Machine Learning Model**:
    - **Model Selection**: Chose **Random Forest** for multilabel classification, as it handles multiple outputs, reduces overfitting, provides feature importance insights, and is scalable with large datasets.
    - **Data Splitting**: Split data into training (80%) and testing (20%) sets using `train_test_split`.
    - **Model Training**: Trained a `RandomForestClassifier` with 100 estimators, using `MultiOutputClassifier` to handle multilabel classification.
Feature Importance Extraction: Averaged feature importances from all trees to identify key features influencing position predictions.

3. **Evaluating the Machine Learning Model**:
    - **Creating Function to Recommend Player Positions**: Defined a `select_and_sort()` function using the model trained in Step 2 to recommend player positions. The function uses the model to calculate suitable probabilities for each position based on player features, sorts the positions by their suitability, and returns a ranked list of positions along with their corresponding probabilities.
    - **Model Accuracy Evaluation**: Evaluated the model's performance by comparing the top recommended position with the player’s actual positions. A recommendation was considered correct if the top recommended position matched any of the player’s actual positions. The accuracy was calculated as the ratio of correct recommendations to the total number of recommendations in a subset of test data.
      
4. **Using the Machine Learning Model for Position Recommendations Based on Player's Input Data**:
    - **Creating Function to Collect and Transform player data from keyboard input**: Defined a `get_player_input()` function to collect player input from the keyboard based on the format provided in [EA Sports FC Official Ratings](https://www.ea.com/games/ea-sports-fc/ratings). The function processes the collected input, transforms it into a format compatible with the model's feature requirements.
    - **Generating Position Recommendations for a Player from Keyboard Input**: Used the `select_and_sort()` function (defined in Step 3) to recommend player positions based on keyboard input.
   
5. **Output Visualization**:
   - **Filtering Unfound Positions**: Identified positions not found in the training dataset by counting occurrences of each position from `pos_list`.
Excluded positions that had zero occurrences from the visual representation, ensuring only relevant positions are visualized.
    - **Visualizing Position Recommendations**:
          - Pitch Setup: Used `matplotlib` to plot a football pitch with a rectangular field, penalty boxes, and a center circle.
          - Position Plotting:
              - Defined player positions using a set of predefined coordinates on the pitch.
              - Defined a function to interpolate custom colors based on position suitability, where positions with higher suitability are given greener shades, while lower suitability                 is represented with warmer colors like red and orange.
Color Coding for Position Suitability:
Top positions (above 0.15 probability) are highlighted in different shades, with brighter colors indicating higher suitability.
Positions with zero probability are marked in red to indicate their irrelevance.
A ranking system is applied, where the most suitable positions are given higher ranks and shown with distinct colors and labels.
Rankings and Labels:

Displayed a ranked list of positions next to the pitch, showing suitability probabilities in percentages for the top recommended positions.
Each position’s ranking was displayed directly on the pitch, highlighting their suitability based on the model’s output.
Visualization Enhancements:

Used plt.Circle and plt.text for plotting player positions and displaying their corresponding names and rankings on the pitch.
The visualization utilized dynamic color mapping, where the color intensity changes based on the probability value of each position.
The rank list was presented in the sidebar of the plot to provide a clear, concise ranking system for the most suitable player positions.
Final Presentation:

Ensured the visualization was clear and informative by adjusting the plot aesthetics (font, color, layout) and using ax for custom plot handling.
Added a border around the pitch and used proper axis adjustments to ensure a professional look.
Titles and labels were added to make the visualization easier to understand and interpret.

6. **Building a Web Interface with Streamlit**:


## Visual Examples
add Image Here

## Future Enhancements

- **Simplified Input Options**: Add a fast entry option for casual users to make the process quicker and easier.
- **Deep Learning Integration**: Implement advanced models to provide more accurate and detailed predictions.
- **Team Analysis**: Expand features to optimize team formations and tactical strategies.
- **Interactive Customization**: Enable real-time adjustment of attribute weights during analysis.


## Acknowledgments

- [Stefano Leone](https://www.kaggle.com/stefanoleone992) for EA Sports FC 24 Complete Player Dataset
- [Davis Nyagami](https://www.kaggle.com/nyagami) for EA SPORTS FC 25 DATABASE, RATINGS AND STATS
- [EA Sports FC Official Ratings](https://www.ea.com/games/ea-sports-fc/ratings) for the input format and player ratings structure.


