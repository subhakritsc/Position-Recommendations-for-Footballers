# Position Recommendations for Footballers Using Python and Scikit-learn

**[My Project Notebook](https://github.com/subhakritsc/Position-Recommendations-for-Footballers/blob/main/Football%20Position%20Selection%20Test.ipynb)**

## Objective

This project aims to help football players and coaches select the most suitable playing position for a player based on their physical attributes and skills. Using machine learning techniques, the model selects and ranks the best positions for a player, assisting in optimal team selection and player development.

## Dataset

The project uses two datasets for model training and testing:

1. **[EA Sports FC 24 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)** by **[Stefano Leone](https://www.kaggle.com/stefanoleone992)**: Used for **training** the model.

2. **[EA SPORTS FC 25 DATABASE, RATINGS AND STATS](https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats)** by **[Davis Nyagami](https://www.kaggle.com/nyagami)**:Used for **testing** the model.

## Key Steps in the Project

1. **Data Preprocessing**:
   - Addressing missing values and outliers to maintain data integrity.
   - Normalizing attributes to ensure uniform scaling for analysis.

2. **Feature Engineering**:
   - Calculating derived metrics like BMI (Body Mass Index) to enhance physical evaluations.
   - Aggregating skills into key metrics (e.g., offensive, defensive, and physical averages) for concise insights.

3. **Model Training and Evaluation**:
   - Training a Random Forest Classifier to predict the most suitable positions for players.
   - Refining the model through cross-validation and hyperparameter tuning for better accuracy.

4. **Result Presentation**:
   - Visualizing position recommendations with charts and graphs for easy interpretation.
   - Highlighting strengths and weaknesses to inform decisions effectively.

## Future Enhancements

- **Simplified Input Options**: Add a fast entry option for casual users to make the process quicker and easier.
- **Deep Learning Integration**: Implement advanced models to provide more accurate and detailed predictions.
- **Team Analysis**: Expand features to optimize team formations and tactical strategies.
- **Interactive Customization**: Enable real-time adjustment of attribute weights during analysis.


## Acknowledgments

- **[Stefano Leone](https://www.kaggle.com/stefanoleone992)** for EA Sports FC 24 Complete Player Dataset
- **[Davis Nyagami](https://www.kaggle.com/nyagami)** for EA SPORTS FC 25 DATABASE, RATINGS AND STATS
- **[EA Sports FC Official Ratings](https://www.ea.com/games/ea-sports-fc/ratings)** for the input format and player ratings structure.


