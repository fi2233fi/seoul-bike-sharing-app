import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🎯 Load trained model (Ensure the file exists in the directory)
MODEL_PATH = "/Users/fatimaiqbal/bike_project/bike_sharing_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    trained_features = model.feature_names_in_
except FileNotFoundError:
    st.error("🚨 Model file not found! Ensure 'bike_sharing_model.pkl' exists.")
    model = None
    trained_features = []

# 🚴‍♂️ Title
st.title("🚲 AI-Powered Bike Rental Prediction & Marketing Insights")

# 📊 Sidebar for user inputs
st.sidebar.header("Enter Bike Rental Conditions")

# 🎛 User Inputs
hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Wind Speed (m/s)", 0, 50, 10)
day = st.sidebar.slider("Day of the Month", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.selectbox("Year", [2011, 2012])
weekday = st.sidebar.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, 3)
is_weekend = 1 if weekday >= 5 else 0  # Weekends = 1 (Saturday, Sunday)

# 🎯 Convert categorical inputs to match model encoding
season_map = {"Spring": 2, "Summer": 3, "Fall": 4, "Winter": 1}
weather_map = {"Clear": 1, "Cloudy": 2, "Light Rain/Snow": 3, "Heavy Rain/Snow": 4}
season = st.sidebar.selectbox("Season", list(season_map.keys()))
weather = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()))
holiday = 1 if st.sidebar.radio("Is it a Holiday?", ["No", "Yes"]) == "Yes" else 0
workingday = 1 if st.sidebar.radio("Is it a Working Day?", ["No", "Yes"]) == "Yes" else 0

# Convert season and weather to one-hot encoded format
season_features = {f"season_{i}": 1 if season_map[season] == i else 0 for i in [2, 3, 4]}
weather_features = {f"weather_{i}": 1 if weather_map[weather] == i else 0 for i in [2, 3, 4]}

# 🔄 Prepare input DataFrame with correct features
input_data = pd.DataFrame([[temp, 0, humidity, windspeed, hour, day, month, year,
                            season_features["season_2"], season_features["season_3"], season_features["season_4"],
                            holiday, workingday,
                            weather_features["weather_2"], weather_features["weather_3"], weather_features["weather_4"],
                            weekday, is_weekend]],
                          columns=trained_features)

# Ensure feature alignment (Fixing Feature Mismatch Issues)
missing_cols = [col for col in trained_features if col not in input_data.columns]
for col in missing_cols:
    input_data[col] = 0  # Add missing columns with zero values

# 🔮 Predict rentals
if model:
    predicted_rentals = model.predict(input_data)[0]
    st.subheader(f"📊 Predicted Bike Rentals: {round(predicted_rentals)}")
else:
    st.error("⚠️ Prediction not available. Model not loaded.")

# 🎯 AI-Powered Marketing Insights
def marketing_message(predictions):
    if predictions > 200:
        return "🚀 High Demand! Consider surge pricing and social media ads."
    elif 100 <= predictions <= 200:
        return "📈 Moderate Demand! Target work commuters with limited-time offers."
    else:
        return "📉 Low Demand! Offer discounts for off-peak hours and family packages."

if "predicted_rentals" in locals():
    message = marketing_message(predicted_rentals)
    st.info(f"💡 Marketing Strategy: {message}")

# 📊 Load training data for visualizations
DATA_PATH = "/Users/fatimaiqbal/bike_project/train.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.lower().str.strip()  # Standardize column names
        return df
    except FileNotFoundError:
        st.error("⚠️ train.csv not found! Please ensure the dataset is available.")
        return None

# ✅ Debugging: Check if dataset loads properly
df = load_data()
if df is not None:
    st.write("📌 **Dataset Shape:**", df.shape)  # Show dataset dimensions
    st.write("📌 **First 5 Rows:**")  
    st.write(df.head())  # Display first few rows
else:
    st.error("⚠️ Dataset not found or empty! Check file path.")

if df is not None:
    # 📈 Visualization: Bike Rentals by Hour
    if "hour" in df.columns and "count" in df.columns:
        st.subheader("📊 Bike Rental Trends")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x="hour", y="count", ax=ax)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Number of Rentals")
        plt.title("Bike Rentals Trend by Hour")
        st.pyplot(fig)
    else:
        st.error("🚨 Required columns 'hour' and 'count' not found! Check dataset.")

    # 🔍 Visualization: Correlation Heatmap
    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

    # 🌦 Bike Rentals by Weather Condition
    if "weather" in df.columns:
        st.subheader("🌦 Bike Rentals by Weather Condition")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="weather", y="count", ax=ax, palette="viridis")
        plt.xlabel("Weather Condition (1 = Clear, 4 = Heavy Rain/Snow)")
        plt.ylabel("Average Bike Rentals")
        plt.title("Impact of Weather on Bike Rentals")
        st.pyplot(fig)

    # 🍂 Bike Rentals by Season
    if "season" in df.columns:
        st.subheader("🍂 Bike Rentals by Season")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="season", y="count", ax=ax, palette="coolwarm")
        plt.xlabel("Season (1 = Winter, 4 = Fall)")
        plt.ylabel("Average Bike Rentals")
        plt.title("Bike Rentals by Season")
        st.pyplot(fig)

    # 📆 Weekends vs. Weekdays Bike Rentals
    if "is_weekend" in df.columns:
        st.subheader("📆 Weekends vs. Weekdays Bike Rentals")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x="is_weekend", y="count", ax=ax, palette="magma")
        plt.xticks(ticks=[0, 1], labels=["Weekday", "Weekend"])
        plt.xlabel("Day Type")
        plt.ylabel("Bike Rentals")
        plt.title("Bike Rentals on Weekdays vs. Weekends")
        st.pyplot(fig)

# 🎉 Final success message
st.success("🎉 AI-Powered Bike Rental Prediction App is Ready!")

