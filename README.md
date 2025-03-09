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

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Select a stock from the dropdown list.
3. Click the **Train and Predict** button to train the model and generate predictions.
4. View the predicted prices and interactive visualization.

## Model Details
The Transformer model consists of:
- An embedding layer to map input features to a higher-dimensional space
- Multiple Transformer encoder layers for feature extraction
- A fully connected layer to generate final predictions

## Data Processing
The following technical indicators are calculated:
- **SMA (50-day), EMA (50-day)**
- **Daily Returns and Log Returns**
- **Relative Strength Index (RSI-14)**
- **Bollinger Bands**
- **Momentum and Volatility**

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments
- Yahoo Finance for stock data
- Streamlit for an easy-to-use web interface
- PyTorch for deep learning implementation

