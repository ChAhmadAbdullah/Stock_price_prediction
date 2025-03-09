# Stock Price Prediction Using Transformer Model

## Overview
This project utilizes a Transformer-based deep learning model to predict stock prices using historical stock data. The model is implemented using PyTorch and Streamlit for an interactive web application. It fetches stock data from Yahoo Finance and applies various technical indicators before training the model.

## Features
- Fetches historical stock data for the past 10 years
- Computes technical indicators such as Moving Averages, RSI, and Bollinger Bands
- Normalizes data and prepares sequences for training
- Implements a Transformer model for time-series forecasting
- Allows users to select stocks and predict prices for the next 10 days
- Visualizes predictions using Plotly
- Cached models for faster prediction

## Technologies Used
- **Python**
- **yfinance** (for stock data retrieval)
- **pandas, numpy** (for data processing)
- **scikit-learn** (for data scaling and splitting)
- **PyTorch** (for the Transformer model)
- **Streamlit** (for UI)
- **Plotly** (for visualization)

## Usage
1. Run the Streamlit app:
2. Select a stock from the dropdown list.
3. Click the **Train and Predict** button to train the model and generate predictions.
4. View the predicted prices and interactive visualization.

## Model Details
The Transformer model consists of:
- An embedding layer to map input features to a higher-dimensional space
- Multiple Transformer encoder layers for feature extraction
- A fully connected layer to generate final predictions


