# SUPERVISED_ML-3
This script explores wind power prediction using linear regression on a dataset containing wind speed and power output information.

Data Loading and Exploration

Import Libraries:

The code imports necessary libraries for numerical operations (NumPy), data manipulation (Pandas), machine learning (scikit-learn), visualization (Seaborn, Matplotlib), and inline plotting (%matplotlib inline).
Load Data:

Three CSV files are loaded using Pandas' read_csv:
train-en.csv: Training data
eval-en.csv: Evaluation data
challenge-en.csv: Challenge data (presumably for future predictions)
Data Exploration:

describe is used to get summary statistics of the training data.
head displays the first few rows of the training data to understand its structure.
All column names are printed using a loop to reveal any truncated information.
info provides detailed information about data types and memory usage.
The relationship between wind speed and power output is explored using groupby and describe on the training data.
Data Preparation

Feature and Target Selection:

Wind speed (wind_speed48M) is selected as the feature (independent variable) for training the model.
Power output (Output) is selected as the target variable (dependent variable) to be predicted.
Data Split (Optional):

While not explicitly shown, the code mentions splitting the training data into training and testing sets for model evaluation. This can be achieved using sklearn.model_selection.train_test_split.
Visualization:

A scatter plot is created using Matplotlib to visualize the relationship between wind speed and power output in the training data.
Model Training and Evaluation

Model Definition and Training:

A linear regression model is created using sklearn.linear_model.LinearRegression.
The model is trained on the selected features and target variables from the training data using .fit.
Model Performance on Training Data:

The model's R-squared score (reg.score) is calculated to evaluate its performance on the training data. R-squared indicates the proportion of variance in the target variable explained by the model.
Prediction on Evaluation Data:

The trained model is used to predict power output for the wind speed values in the evaluation data using .predict.
Visualization of Predictions:

Another scatter plot is created to visualize the actual power output (from the evaluation data) versus the predicted power output.
Correlation Analysis (Optional):

The correlation matrix for the training data can be calculated using .corr() to explore relationships between features. The script demonstrates calculating the correlation specifically between wind speed and power output.
Conclusion

This script demonstrates the application of linear regression for wind power prediction. It covers data loading, exploration, preparation, model training, evaluation, and visualization. By analyzing the R-squared score and the scatter plots, you can assess the model's ability to learn the relationship between wind speed and power output.
