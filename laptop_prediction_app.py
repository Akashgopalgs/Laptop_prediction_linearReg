import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the trained model
try:
    with open('laptop_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'laptop_price_model.pkl' is in the correct directory.")
    st.stop()

# Load feature names
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Define scalers based on your trained model
scaler_x = StandardScaler()
scaler_y = MinMaxScaler()

# Function to scale input data
def scale_input_data(data):
    data[['x resolution']] = scaler_x.fit_transform(data[['x resolution']])
    data[['y resolution']] = scaler_x.fit_transform(data[['y resolution']])
    data[['Inches']] = scaler_y.fit_transform(data[['Inches']])
    data[['Ram']] = scaler_y.fit_transform(data[['Ram']])
    data[['Weight']] = scaler_y.fit_transform(data[['Weight']])
    data[['storage']] = scaler_y.fit_transform(data[['storage']])
    return data

# Function to convert storage to GB
def convert_to_gb(storage_str):
    amounts = re.findall(r'(\d*\.?\d+)([TtGgBb]+)', storage_str)
    total_gb = 0
    for amount, unit in amounts:
        amount = float(amount)
        if 'TB' in unit.upper():
            total_gb += amount * 1024
        elif 'GB' in unit.upper():
            total_gb += amount
    return total_gb

# App title
st.title("Laptop Price Prediction App")

# Input fields using select boxes
company = st.selectbox('Company',
                       ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI', 'Microsoft', 'Toshiba',
                        'Huawei', 'Xiaomi', 'Vero', 'Razer', 'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG'])
typename = st.selectbox('Type', ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', 'Convertible', 'Workstation'])
os = st.selectbox('Operating System', ['macOS', 'No OS', 'Windows', 'Linux', 'Android', 'Chrome OS'])
inches = st.selectbox('Inches', np.arange(10.0, 20.1, 1))
ram = st.selectbox('Ram (GB)', [4, 8, 16, 32, 64])
weight = st.selectbox('Weight (kg)', np.arange(0.5, 5.1, 1))
screen_resolution = st.selectbox('Screen Resolution',
                                 ['1366x768', '1600x900', '1920x1080', '2560x1440', '2560x1600', '3840x2160'])
cpu = st.selectbox('CPU', ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD', 'Other'])
gpu = st.selectbox('GPU', ['Intel HD Graphics 620', 'AMD Radeon', 'Nvidia GeForce', 'Other'])
memory = st.selectbox('Memory', ['128GB SSD', '256GB SSD', '512GB SSD', '1TB HDD', '2TB HDD', '256GB Flash Storage'])

# Button to predict
if st.button('Predict Price'):
    # Process the input data
    data = {
        'Company': [company],
        'TypeName': [typename],
        'OpSys': [os],
        'Inches': [inches],
        'Ram': [ram],
        'Weight': [weight],
        'ScreenResolution': [screen_resolution],
        'Cpu': [cpu],
        'Gpu': [gpu],
        'Memory': [memory]
    }
    input_data = pd.DataFrame(data)

    # Extract features from ScreenResolution
    input_data['IPS Panel'] = input_data['ScreenResolution'].str.contains('IPS Panel').astype(int)
    input_data['Touchscreen'] = input_data['ScreenResolution'].str.contains('Touchscreen').astype(int)
    input_data['x resolution'] = input_data['ScreenResolution'].str.extract(r'(\d+)x').astype(int)
    input_data['y resolution'] = input_data['ScreenResolution'].str.extract(r'x(\d+)').astype(int)
    input_data.drop(columns=['ScreenResolution'], inplace=True)

    # Extract features from CPU and GPU
    input_data['intel i3'] = input_data['Cpu'].str.contains('Intel Core i3').astype(int)
    input_data['intel i5'] = input_data['Cpu'].str.contains('Intel Core i5').astype(int)
    input_data['intel i7'] = input_data['Cpu'].str.contains('Intel Core i7').astype(int)
    input_data['AMD'] = input_data['Cpu'].str.contains('AMD').astype(int)
    input_data['other cpu'] = (~input_data['Cpu'].str.contains('Intel Core i3|Intel Core i5|Intel Core i7|AMD')).astype(int)
    input_data['intel gpu'] = input_data['Gpu'].str.contains('Intel').astype(int)
    input_data['AMD gpu'] = input_data['Gpu'].str.contains('AMD').astype(int)
    input_data['Nvidia'] = input_data['Gpu'].str.contains('Nvidia').astype(int)
    input_data.drop(['Cpu', 'Gpu'], axis=1, inplace=True)

    # Extract features from Memory
    input_data['SSD'] = input_data['Memory'].str.contains('SSD').astype(int)
    input_data['HDD'] = input_data['Memory'].str.contains('HDD').astype(int)
    input_data['Flash Storage'] = input_data['Memory'].str.contains('Flash Storage').astype(int)
    input_data['Hybrid'] = input_data['Memory'].str.contains('Hybrid').astype(int)
    input_data['storage'] = input_data['Memory'].apply(lambda x: convert_to_gb(x))
    input_data.drop(columns=['Memory'], inplace=True)

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, columns=['Company', 'TypeName', 'OpSys'], drop_first=True)

    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match training data
    input_data = input_data[feature_names]

    # Scale the input data
    input_data = scale_input_data(input_data)

    # Predict the price
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"##### The predicted price of the laptop is: ${prediction[0]:.2f}")
