import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
st.set_page_config(
    page_title="Seattle Weather Analytics Dashboard",
    page_icon="🌦️",
    layout="wide"
)
st.title("🌦️ Seattle Weather Analytics Dashboard")
st.write(
    "Weather Data Analysis, Visualization and Prediction"
)
# LOAD DATA
@st.cache_data
def load_data():
    data = pd.read_csv(
        "cleaned_seattle_waether.csv"
    )
    return data
data = load_data()
# DATA PREPARATION
data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)
# CREATE DATE COLUMNS
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["month_name"] = data["date"].dt.month_name()
data["day"] = data["date"].dt.day
# SIDEBAR
st.sidebar.header("🎛️ Weather Prediction")
st.sidebar.write(
    "Enter weather conditions manually and click "
    "Predict Weather."
)
# PRECIPITATION INPUT
precipitation_input = st.sidebar.number_input(
    "🌧️ Precipitation",
    min_value=float(data["precipitation"].min()),
    max_value=float(data["precipitation"].max()),
    value=float(data["precipitation"].mean()),
    step=0.1
)
# MAX TEMPERATURE INPUT
temp_max_input = st.sidebar.number_input(
    "🌡️ Maximum Temperature",
    min_value=float(data["temp_max"].min()),
    max_value=float(data["temp_max"].max()),
    value=float(data["temp_max"].mean()),
    step=0.1
)
# MIN TEMPERATURE INPUT
temp_min_input = st.sidebar.number_input(
    "❄️ Minimum Temperature",
    min_value=float(data["temp_min"].min()),
    max_value=float(data["temp_min"].max()),
    value=float(data["temp_min"].mean()),
    step=0.1
)
# WIND INPUT
wind_input = st.sidebar.number_input(
    "💨 Wind Speed",
    min_value=float(data["wind"].min()),
    max_value=float(data["wind"].max()),
    value=float(data["wind"].mean()),
    step=0.1
)
# MONTH INPUT
month_input = st.sidebar.selectbox(
    "📅 Month",
    list(range(1, 13)),
    index=0
)
# PREDICT BUTTON
predict_button = st.sidebar.button(
    "🔮 Predict Weather"
)
# DASHBOARD HEADER
st.header("📊 Weather Overview")
# KPI CALCULATIONS
total_records = len(data)
average_precipitation = (
    data["precipitation"].mean()
)
average_temp_max = (
    data["temp_max"].mean()
)
average_temp_min = (
    data["temp_min"].mean()
)
average_wind = (
    data["wind"].mean()
)
# KPI CARDS
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "📋 Total Records",
        f"{total_records:,}"
    )
with col2:
    st.metric(
        "🌧️ Avg Precipitation",
        f"{average_precipitation:.2f}"
    )
with col3:
    st.metric(
        "🌡️ Avg Max Temperature",
        f"{average_temp_max:.2f}°C"
    )
with col4:
    st.metric(
        "💨 Avg Wind",
        f"{average_wind:.2f}"
    )
# WEATHER DISTRIBUTION
st.header("🌦️ Weather Distribution")
weather_count = (
    data["weather"]
    .value_counts()
    .reset_index()
)
weather_count.columns = [
    "weather",
    "count"
]
fig_weather = px.bar(
    weather_count,
    x="weather",
    y="count",
    title="Number of Days by Weather Type",
    text="count",
    color="weather",
    color_discrete_map={
        "rain": "#3498DB",
        "sun": "#F1C40F",
        "fog": "#95A5A6",
        "drizzle": "#1ABC9C",
        "snow": "#9B59B6"
    }
)
fig_weather.update_layout(
    xaxis_title="Weather",
    yaxis_title="Number of Days",
    height=420
)
st.plotly_chart(
    fig_weather,
    width="stretch"
)
# MONTHLY PRECIPITATION
st.header("📈 Weather Trend Analysis")
monthly_weather = (
    data
    .groupby("month")["precipitation"]
    .mean()
    .reset_index()
)
fig_month = px.line(
    monthly_weather,
    x="month",
    y="precipitation",
    markers=True,
    title="Average Monthly Precipitation"
)
fig_month.update_traces(
    line=dict(
        color="#00A8A8",
        width=3
    ),
    marker=dict(
        color="#00A8A8",
        size=8
    )
)
fig_month.update_layout(
    xaxis_title="Month",
    yaxis_title="Average Precipitation",
    height=420
)
st.plotly_chart(
    fig_month,
    width="stretch"
)
# TEMPERATURE AND WIND
col1, col2 = st.columns(2)
# MONTHLY TEMPERATURE
with col1:
    monthly_temp = (
        data
        .groupby("month")[
            [
                "temp_max",
                "temp_min"
            ]
        ]
        .mean()
        .reset_index()
    )
    temp_long = monthly_temp.melt(
        id_vars="month",
        value_vars=[
            "temp_max",
            "temp_min"
        ],
        var_name="Temperature Type",
        value_name="Temperature"
    )
    fig_temp = px.line(
        temp_long,
        x="month",
        y="Temperature",
        color="Temperature Type",
        markers=True,
        title="Monthly Temperature Trend",
        color_discrete_map={
            "temp_max": "#E74C3C",
            "temp_min": "#3498DB"
        }
    )
    fig_temp.update_layout(
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        height=420
    )


    st.plotly_chart(
        fig_temp,
        width="stretch"
    )
# WIND SPEED
with col2:
    monthly_wind = (
        data
        .groupby("month")["wind"]
        .mean()
        .reset_index()
    )
    fig_wind = px.bar(
        monthly_wind,
        x="month",
        y="wind",
        title="Average Wind Speed by Month",
        color="wind",
        color_continuous_scale="Purples"
    )
    fig_wind.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Wind Speed",
        height=420
    )
    st.plotly_chart(
        fig_wind,
        width="stretch"
    )
# TEMPERATURE VS PRECIPITATION
col1, col2 = st.columns(2)
with col1:
    fig_precip_temp = px.scatter(
        data,
        x="temp_max",
        y="precipitation",
        title="Maximum Temperature vs Precipitation",
        opacity=0.7,
        color="precipitation",
        color_continuous_scale="Greens"
    )
    fig_precip_temp.update_layout(
        xaxis_title="Maximum Temperature",
        yaxis_title="Precipitation",
        height=420
    )
    st.plotly_chart(
        fig_precip_temp,
        width="stretch"
    )
# WIND VS PRECIPITATION
with col2:
    fig_wind_precip = px.scatter(
        data,
        x="wind",
        y="precipitation",
        title="Wind Speed vs Precipitation",
        opacity=0.7,
        color="wind",
        color_continuous_scale="Reds"
    )
    fig_wind_precip.update_layout(
        xaxis_title="Wind Speed",
        yaxis_title="Precipitation",
        height=420
    )
    st.plotly_chart(
        fig_wind_precip,
        width="stretch"
    )
# WEATHER TYPE BY MONTH
st.header("📅 Weather Type by Month")
weather_month = pd.crosstab(
    data["month"],
    data["weather"]
).reset_index()
weather_month_long = weather_month.melt(
    id_vars="month",
    var_name="Weather",
    value_name="Number of Days"
)
fig_weather_month = px.bar(
    weather_month_long,
    x="month",
    y="Number of Days",
    color="Weather",
    title="Weather Types Across Months",
    barmode="stack",
    color_discrete_map={
        "rain": "#3498DB",
        "sun": "#F1C40F",
        "fog": "#95A5A6",
        "drizzle": "#1ABC9C",
        "snow": "#9B59B6"
    }
)
fig_weather_month.update_layout(
    xaxis_title="Month",
    yaxis_title="Number of Days",
    height=450
)
st.plotly_chart(
    fig_weather_month,
    width="stretch"
)
# STATISTICAL SUMMARY
st.header("📊 Statistical Summary")
statistics = data[
    [
        "precipitation",
        "temp_max",
        "temp_min",
        "wind"
    ]
].describe()
st.dataframe(
    statistics,
    width="stretch"
)
# MACHINE LEARNING
st.header(
    "🤖 Machine Learning - Logistic Regression"
)
st.write(
    "The model predicts the weather type based on "
    "precipitation, temperature, wind and month."
)
# WEATHER CLASS DISTRIBUTION
st.subheader(
    "🔍 Weather Class Distribution"
)
target_count = (
    data["weather"]
    .value_counts()
    .reset_index()
)
target_count.columns = [
    "Weather",
    "Number of Records"
]
st.dataframe(
    target_count,
    width="stretch",
    hide_index=True
)
# SELECT FEATURES
X = data[
    [
        "precipitation",
        "temp_max",
        "temp_min",
        "wind",
        "month"
    ]
]
# TARGET
y = data[
    "weather"
]
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train
)
X_test_scaled = scaler.transform(
    X_test
)
# LOGISTIC REGRESSION
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
# TRAIN MODEL
with st.spinner(
    "Training Logistic Regression model..."
):

    model.fit(
        X_train_scaled,
        y_train
    )
# TEST PREDICTION
test_prediction = model.predict(
    X_test_scaled
)
# MODEL METRICS
accuracy = accuracy_score(
    y_test,
    test_prediction
)
precision = precision_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    test_prediction,
    average="weighted",
    zero_division=0
)
# MODEL PERFORMANCE
st.subheader(
    "🎯 Model Performance"
)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )
with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )
with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )
with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )
# CONFUSION MATRIX
st.subheader(
    "🔢 Confusion Matrix"
)
weather_classes = sorted(
    y.unique()
)
matrix = confusion_matrix(
    y_test,
    test_prediction,
    labels=weather_classes
)
matrix_data = pd.DataFrame(
    matrix,
    index=[
        "Actual " + str(x)
        for x in weather_classes
    ],
    columns=[
        "Predicted " + str(x)
        for x in weather_classes
    ]
)
st.dataframe(
    matrix_data,
    width="stretch"
)
# MANUAL PREDICTION
st.header(
    "🔮 Manual Weather Prediction"
)
st.write(
    "Change the values in the sidebar and click "
    "'Predict Weather'. The prediction will appear here."
)
# PREDICTION
if predict_button:
    # CREATE MANUAL INPUT
    manual_data = pd.DataFrame(
        [
            [
                precipitation_input,
                temp_max_input,
                temp_min_input,
                wind_input,
                month_input
            ]
        ],
        columns=[
            "precipitation",
            "temp_max",
            "temp_min",
            "wind",
            "month"
        ]
    )
    # SCALE MANUAL INPUT
    manual_data_scaled = scaler.transform(
        manual_data
    )
    # PREDICT WEATHER
    manual_prediction = model.predict(
        manual_data_scaled
    )
    predicted_weather = manual_prediction[0]
    # PREDICTION PROBABILITY
    prediction_probabilities = (
        model.predict_proba(
            manual_data_scaled
        )[0]
    )
    probability_data = pd.DataFrame(
        {
            "Weather": model.classes_,
            "Probability": (
                prediction_probabilities * 100
            )
        }
    )
    probability_data = (
        probability_data
        .sort_values(
            "Probability",
            ascending=False
        )
    )
    # DISPLAY RESULT
    st.subheader(
        "🎯 Prediction Result"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.success(
            f"🌦️ Predicted Weather: "
            f"{predicted_weather.upper()}"
        )
    with col2:
        highest_probability = (
            prediction_probabilities.max()
        )
        st.metric(
            "Prediction Confidence",
            f"{highest_probability * 100:.2f}%"
        )
    # PROBABILITY CHART
    st.subheader(
        "📊 Weather Prediction Probabilities"
    )
    fig_probability = px.bar(
        probability_data,
        x="Weather",
        y="Probability",
        text="Probability",
        title="Probability of Each Weather Type",
        color="Weather",
        color_discrete_map={
            "rain": "#3498DB",
            "sun": "#F1C40F",
            "fog": "#95A5A6",
            "drizzle": "#1ABC9C",
            "snow": "#9B59B6"
        }
    )
    fig_probability.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig_probability.update_layout(
        xaxis_title="Weather Type",
        yaxis_title="Probability (%)",
        height=420
    )
    st.plotly_chart(
        fig_probability,
        width="stretch"
    )
    # INPUT VALUES
    st.subheader(
        "📝 Entered Values"
    )
    input_display = pd.DataFrame(
        {
            "Parameter": [
                "Precipitation",
                "Maximum Temperature",
                "Minimum Temperature",
                "Wind Speed",
                "Month"
            ],
            "Value": [
                precipitation_input,
                temp_max_input,
                temp_min_input,
                wind_input,
                month_input
            ]
        }
    )
    st.dataframe(
        input_display,
        width="stretch",
        hide_index=True
    )
# FOOTER
st.divider()
st.caption(
    "Seattle Weather Analytics Dashboard | "
    "Python • Pandas • Plotly • Streamlit • Scikit-learn"
)
