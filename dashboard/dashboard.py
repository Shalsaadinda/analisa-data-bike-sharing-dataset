import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title of the Streamlit app
st.title("Bike Sharing Data Analysis")

# Specify the paths to your CSV files
day_file_path = 'data/day.csv'  # Change this path if needed
hour_file_path = 'data/hour.csv'  # Change this path if needed

# Load the data
day_data = pd.read_csv(day_file_path)
hour_data = pd.read_csv(hour_file_path)

# Displaying first few rows of the data
st.subheader("Data from 'day.csv'")
st.write(day_data.head())

st.subheader("Data from 'hour.csv'")
st.write(hour_data.head())

# General information about the datasets
st.subheader("General Information about 'day.csv'")
st.write(day_data.info())

st.subheader("Descriptive Statistics for 'day.csv'")
st.write(day_data.describe())

st.subheader("General Information about 'hour.csv'")
st.write(hour_data.info())

st.subheader("Descriptive Statistics for 'hour.csv'")
st.write(hour_data.describe())

# Display insights
st.subheader("Insights from 'day.csv'")
st.write(f"Number of columns: {day_data.shape[1]}")
st.write(f"Number of rows: {day_data.shape[0]}")
st.write(f"Columns with null values:\n{day_data.isnull().sum()}")

st.subheader("Insights from 'hour.csv'")
st.write(f"Number of columns: {hour_data.shape[1]}")
st.write(f"Number of rows: {hour_data.shape[0]}")
st.write(f"Columns with null values:\n{hour_data.isnull().sum()}")

# Check for duplicates and clean data if necessary
if day_data.duplicated().sum() > 0:
    day_data = day_data.drop_duplicates()
    st.write(f"Duplicates found and removed from 'day.csv'")

if hour_data.duplicated().sum() > 0:
    hour_data = hour_data.drop_duplicates()
    st.write(f"Duplicates found and removed from 'hour.csv'")

# Correlation heatmap for day.csv
st.subheader("Correlation Heatmap for 'day.csv'")
numerical_cols_day = day_data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_cols_day.corr(), annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={"size": 10}, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title('Correlation Matrix - day.csv', fontsize=12)
plt.tight_layout()
st.pyplot(plt.gcf())

# Correlation heatmap for hour.csv
st.subheader("Correlation Heatmap for 'hour.csv'")
numerical_cols_hour = hour_data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numerical_cols_hour.corr(), annot=True, cmap='coolwarm', fmt='.2f',
            annot_kws={"size": 10}, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title('Correlation Matrix - hour.csv', fontsize=12)
plt.tight_layout()
st.pyplot(plt.gcf())

# Scatter plot of windspeed vs bike count
st.subheader("Scatter Plot: Windspeed vs Bike Count per Hour")
plt.figure(figsize=(10, 6))
sc = plt.scatter(hour_data['windspeed'], hour_data['cnt'], c=hour_data['cnt'], cmap='Blues', alpha=0.7)
plt.title('Windspeed vs Bike Count per Hour', fontsize=15)
plt.xlabel('Windspeed', fontsize=12)
plt.ylabel('Bike Count', fontsize=12)
cbar = plt.colorbar(sc)
cbar.set_label('Bike Count')
plt.grid()
st.pyplot(plt.gcf())

# Scatter plot of temperature vs bike count
st.subheader("Scatter Plot: Temperature vs Bike Count per Day")
plt.figure(figsize=(10, 6))
sc = plt.scatter(day_data['temp'], day_data['cnt'], c=day_data['cnt'], cmap='Greens', alpha=0.7)
plt.title('Temperature vs Bike Count per Day', fontsize=15)
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Bike Count', fontsize=12)
cbar = plt.colorbar(sc)
cbar.set_label('Bike Count')
plt.grid()
st.pyplot(plt.gcf())
