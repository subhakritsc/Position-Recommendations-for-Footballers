# Position Recommendations for Footballers Using Python and Scikit-Learn


**[My Project Notebook](https://github.com/subhakritsc/Position-Recommendations-for-Footballers/blob/main/Football%20Position%20Selection%20Test.ipynb)**


## Objective

This project aims to help football players and coaches select the most suitable playing position for a player based on their physical attributes and skills. Using machine learning techniques, the model selects and ranks the best positions for a player, assisting in optimal team selection and player development.


## Dataset

The project uses two datasets for model training and testing:

1. **[EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)** by **[Stefano Leone](https://www.kaggle.com/stefanoleone992)**: Used for **Training** the model.

2. **[EA SPORTS FC 25 DATABASE, RATINGS AND STATS](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats)** by **[Davis Nyagami](https://www.kaggle.com/nyagami)**: Used for **Testing** the model.


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
    - **Creating Function to Select and Rank Player Positions**: Developed a function using the model trained in Step 2 to select and rank player positions with probabilities based on features.
    - **Model Accuracy Evaluation**: Compared the top recommended position with the player’s actual positions. If the top recommended position matched one of the player’s actual positions, it was counted as a correct selection. Accuracy was then calculated based on the number of correct selections from a subset of test data.
      
4. **Using the Machine Learning Model for Position Recommendations Based on Player's Input Data**:

5. **Output Visualization**:
   

6. **Building a Web Interface with Streamlit**:


## Visual Examples
add Image Here

## Future Enhancements

- **Simplified Input Options**: Add a fast entry option for casual users to make the process quicker and easier.
- **Deep Learning Integration**: Implement advanced models to provide more accurate and detailed predictions.
- **Team Analysis**: Expand features to optimize team formations and tactical strategies.
- **Interactive Customization**: Enable real-time adjustment of attribute weights during analysis.


## Acknowledgments

- **[Stefano Leone](https://www.kaggle.com/stefanoleone992)** for EA Sports FC 24 Complete Player Dataset
- **[Davis Nyagami](https://www.kaggle.com/nyagami)** for EA SPORTS FC 25 DATABASE, RATINGS AND STATS
- **[EA Sports FC Official Ratings](https://www.ea.com/games/ea-sports-fc/ratings)** for the input format and player ratings structure.


