import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

# 🎯 Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "bike_sharing_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "train.csv")

# 🔁 Load or Train Model
try:
    model = joblib.load(MODEL_PATH)
    trained_features = model.feature_names_in_
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.warning("🚨 Model not found. Training a new one using train.csv...")
    try:
        df_train = pd.read_csv(DATA_PATH)
        df_train.columns = df_train.columns.str.lower().str.strip()

        # Extract features from datetime if needed
        if 'datetime' in df_train.columns:
            df_train['datetime'] = pd.to_datetime(df_train['datetime'])
            df_train['hour'] = df_train['datetime'].dt.hour
            df_train['day'] = df_train['datetime'].dt.day
            df_train['month'] = df_train['datetime'].dt.month
            df_train['year'] = df_train['datetime'].dt.year
            df_train['weekday'] = df_train['datetime'].dt.weekday

        df_train = df_train.rename(columns={"cnt": "count"})
        df_train = df_train.dropna()
        df_train["is_weekend"] = df_train["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # One-hot encoding
        season_dummies = pd.get_dummies(df_train["season"], prefix="season", drop_first=True)
        weather_dummies = pd.get_dummies(df_train["weather"], prefix="weather", drop_first=True)

        features = pd.concat([
            df_train[["temp", "atemp", "humidity", "windspeed", "hour", "day", "month", "year",
                      "holiday", "workingday", "weekday", "is_weekend"]],
            season_dummies,
            weather_dummies
        ], axis=1)

        target = df_train["count"]

        model = LinearRegression()
        model.fit(features, target)
        joblib.dump(model, MODEL_PATH)
        st.success("✅ New model trained and saved!")
        trained_features = model.feature_names_in_
    except Exception as e:
        st.error(f"⚠️ Model training failed: {e}")
        model = None
        trained_features = []

# 🚴‍♂️ Title
st.title("🚲 AI-Powered Bike Rental Prediction & Marketing Insights")

# 📊 Sidebar Inputs
st.sidebar.header("Enter Bike Rental Conditions")

hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
temp = st.sidebar.slider("Temperature (°C)", -10, 40, 20)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Wind Speed (m/s)", 0, 50, 10)
day = st.sidebar.slider("Day of the Month", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.selectbox("Year", [2011, 2012])
weekday = st.sidebar.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, 3)
is_weekend = 1 if weekday >= 5 else 0

season_map = {"Spring": 2, "Summer": 3, "Fall": 4, "Winter": 1}
weather_map = {"Clear": 1, "Cloudy": 2, "Light Rain/Snow": 3, "Heavy Rain/Snow": 4}
season = st.sidebar.selectbox("Season", list(season_map.keys()))
weather = st.sidebar.selectbox("Weather Condition", list(weather_map.keys()))
holiday = 1 if st.sidebar.radio("Is it a Holiday?", ["No", "Yes"]) == "Yes" else 0
workingday = 1 if st.sidebar.radio("Is it a Working Day?", ["No", "Yes"]) == "Yes" else 0

season_features = {f"season_{i}": 1 if season_map[season] == i else 0 for i in [2, 3, 4]}
weather_features = {f"weather_{i}": 1 if weather_map[weather] == i else 0 for i in [2, 3, 4]}

if model:
    input_data = pd.DataFrame([[temp, 0, humidity, windspeed, hour, day, month, year,
                                holiday, workingday, weekday, is_weekend,
                                season_features["season_2"], season_features["season_3"], season_features["season_4"],
                                weather_features["weather_2"], weather_features["weather_3"], weather_features["weather_4"]]],
                              columns=trained_features)

    # Fix missing columns
    for col in trained_features:
        if col not in input_data.columns:
            input_data[col] = 0

    predicted_rentals = model.predict(input_data)[0]
    st.subheader(f"📊 Predicted Bike Rentals: {round(predicted_rentals)}")

    def marketing_message(predictions):
        if predictions > 200:
            return "🚀 High Demand! Consider surge pricing and social media ads."
        elif 100 <= predictions <= 200:
            return "📈 Moderate Demand! Target work commuters with limited-time offers."
        else:
            return "📉 Low Demand! Offer discounts for off-peak hours and family packages."

    st.info(f"💡 Marketing Strategy: {marketing_message(predicted_rentals)}")

# 📥 Load Data for Visuals
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.lower().str.strip()

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['hour'] = df['datetime'].dt.hour
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['weekday'] = df['datetime'].dt.weekday

        if 'cnt' in df.columns and 'count' not in df.columns:
            df.rename(columns={'cnt': 'count'}, inplace=True)

        return df
    except FileNotFoundError:
        st.error("⚠️ train.csv not found! Please ensure the dataset is available.")
        return None

df = load_data()

# 📊 Visualizations
if df is not None:
    st.write("📌 **Dataset Shape:**", df.shape)
    st.write("📌 **First 5 Rows:**")
    st.write(df.head())

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

    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

    if "weather" in df.columns:
        st.subheader("🌦 Bike Rentals by Weather Condition")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="weather", y="count", ax=ax, palette="viridis")
        plt.xlabel("Weather Condition (1 = Clear, 4 = Heavy Rain/Snow)")
        plt.ylabel("Average Bike Rentals")
        plt.title("Impact of Weather on Bike Rentals")
        st.pyplot(fig)

    if "season" in df.columns:
        st.subheader("🍂 Bike Rentals by Season")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="season", y="count", ax=ax, palette="coolwarm")
        plt.xlabel("Season (1 = Winter, 4 = Fall)")
        plt.ylabel("Average Bike Rentals")
        plt.title("Bike Rentals by Season")
        st.pyplot(fig)

