 Seattle Weather Analytics Dashboard
A Streamlit-based weather analytics dashboard for exploring Seattle weather data, visualizing weather trends, and predicting weather types using Logistic Regression.
Project Overview
This project analyzes Seattle weather data using Python and provides an interactive dashboard built with Streamlit.
The dashboard includes:
- Weather data analysis and visualization
- Key weather statistics and KPIs
- Weather distribution by type
- Monthly precipitation trends
- Monthly temperature trends
- Monthly wind-speed analysis
- Temperature vs. precipitation analysis
- Wind speed vs. precipitation analysis
- Weather type distribution by month
- Statistical summary of weather variables
- Logistic Regression-based weather prediction
- Model performance metrics
- Confusion matrix
- Manual weather prediction using user-entered values
- Prediction probability visualization
Technologies Used
- Python
- Streamlit
- Pandas
- Plotly
- Scikit-learn
Machine Learning
The project uses Logistic Regression to predict the weather type.
Input Features
Precipitation
Maximum temperature
Minimum temperature
Wind speed
Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Manual Weather Prediction
Users can enter:
Precipitation
Maximum temperature
Minimum temperature
Wind speed
Month
After clicking Predict Weather, the application displays:
Predicted weather type
Prediction confidence
Probability of each weather type
Entered input values
