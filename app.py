import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Cache for models
model_cache = {}

# Function to fetch stock data
def fetch_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="10y")
    return data[['Close']]

# Function to preprocess stock data
def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df

# Function to normalize data
def normalize_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

# Function to prepare sequences for training
def prepare_data(scaled_data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

# PyTorch Transformer Model
class StockPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)
        src = self.transformer(src)
        output = self.fc(src[-1])  # Use last time step output
        return output

# Function to build the model
def build_model(input_dim):
    model = StockPriceTransformer(input_dim)
    return model

# Streamlit UI
st.title('Stock Price Prediction')

def get_sp500_tickers():
    """Fetch S&P 500 stock tickers from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_stocks = pd.read_html(url)[0]  # Read the first table
    return sp500_stocks["Symbol"].tolist()

# Get a list of stock tickers dynamically
stock_list = get_sp500_tickers()  

# Fallback if fetching fails
if not stock_list:
    stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NFLX', 'NVDA', 'META', 'BABA', 'V', 'JPM', 'DIS', 'WMT']

# Streamlit UI
selected_stock = st.selectbox('Select the Stock:', stock_list)
st.write(f"Fetching data for {selected_stock}...")
# stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
# selected_stock = st.selectbox('Select the Stock:', stock_list)

# st.write(f"Fetching data for {selected_stock}...")
data = fetch_data(selected_stock)
latest_price = data['Close'].iloc[-1]
st.write(f'Latest Price: ${latest_price:.2f}')

if st.button("Train and Predict"):
    st.write("Please Wait...")

    if selected_stock in model_cache:
        model, scaler = model_cache[selected_stock]
    else:
        data = preprocess_data(data)
        scaled_data, scaler = normalize_data(data)

        X, y = prepare_data(scaled_data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        x_train_torch = torch.tensor(x_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        x_test_torch = torch.tensor(x_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        model = build_model(input_dim=1)
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        criterion = nn.MSELoss()

        # Training loop
        epochs = 5
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_train_torch.transpose(0, 1))
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        with torch.no_grad():
            test_outputs = model(x_test_torch.transpose(0, 1))
            test_loss = criterion(test_outputs, y_test_torch)

        st.write(f"Model Evaluation - MSE: {test_loss.item():.4f}")

        model_cache[selected_stock] = (model, scaler)

    st.write("Predicting prices for the next 10 days...")
    predictions = []
    input_sequence = scaled_data[-60:]

    for day in range(10):
        input_tensor = torch.tensor(input_sequence.reshape(60, 1, 1), dtype=torch.float32)
        
        with torch.no_grad():
            predicted_price = model(input_tensor).item()
        
        predictions.append(predicted_price)
        input_sequence = np.append(input_sequence[1:], [[predicted_price]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[:, 0]

    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime('%Y-%m-%d').tolist()
    prediction_df = pd.DataFrame({'Date': days, "Predicted Price": predictions})

    st.write("Predicted Prices:")
    st.table(prediction_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode='lines+markers',
        name='Predicted Prices'
    ))
    fig.update_layout(
        title=f"10-Days Price Prediction for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)