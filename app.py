# Save this entire script as 'app.py'

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import datetime

# --- 1. Configuration and Loading Resources ---
st.set_page_config(layout="wide")
st.title("📈 LSTM Stock Price Trend Predictor")

# Define parameters used during training
TIME_STEP = 60
INPUT_FEATURES = 3 # (Close, SMA_20, RSI)

try:
    # Load the model (using .keras for the updated file, but .h5 works too)
    model = load_model('lstm_model_multi.keras') 
    # Load the scaler object
    scaler = joblib.load('scaler_multi.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.warning("Please ensure 'lstm_model_multi.keras' and 'scaler_multi.pkl' are in the same directory.")
    st.stop()

# --- 2. Feature Engineering Helper Function (Same as training) ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 3. Sidebar and User Input ---
st.sidebar.header("Prediction Settings")
# Default to a popular ticker for ease of use
ticker = st.sidebar.text_input("Enter Stock Ticker:", value='TSLA').upper()
st.sidebar.markdown("---")


# --- 4. Main Prediction Function ---
def predict_next_day_price(ticker):
    # Fetch 200 days of data to guarantee 60-day window and indicator calculation
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=200)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty or len(data) < TIME_STEP:
        st.error(f"Could not fetch enough data for {ticker}. Need at least {TIME_STEP} days.")
        return None, None
    
    df = data.copy()
    
    # Calculate Features
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    
    # Drop NaNs created by indicators
    df = df.dropna()
    
    # Get the last TIME_STEP (60) days of data
    features = ['Close', 'SMA_20', 'RSI']
    input_data = df[features].tail(TIME_STEP)

    if len(input_data) < TIME_STEP:
        st.error(f"After calculating indicators, not enough data remains for a {TIME_STEP}-day window.")
        return None, None
        
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_data)
    
    # Reshape for LSTM: (1 sample, 60 timesteps, 3 features)
    X_predict = np.reshape(scaled_input, (1, TIME_STEP, INPUT_FEATURES))
    
    # Predict (output is scaled)
    scaled_prediction = model.predict(X_predict, verbose=0)
    
    # Inverse Transform: Create a dummy array to rescale the single prediction
    prediction_copies = np.zeros(shape=(1, scaler.n_features_in_))
    prediction_copies[0, 0] = scaled_prediction[0, 0] 
    
    final_prediction = scaler.inverse_transform(prediction_copies)[0, 0]
    
    return final_prediction, df['Close']


# --- 5. Execution and Display ---

if st.sidebar.button("Predict Next Day Price"):
    
    if not ticker:
        st.error("Please enter a stock ticker symbol.")
    else:
        with st.spinner(f"Predicting next day's price for {ticker}..."):
            prediction, close_prices = predict_next_day_price(ticker)
            
            if prediction is not None:
                st.success(f"Prediction Complete!")
                
                # Find the next business day (simple check)
                last_date = close_prices.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                while next_date.weekday() > 4: # 5=Saturday, 6=Sunday
                    next_date += pd.Timedelta(days=1)
                
                # Display Prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Next Predicted Closing Price ({next_date.strftime('%Y-%m-%d')})", 
                              value=f"${prediction:.2f}")
                with col2:
                    st.metric(label=f"Last Recorded Closing Price ({last_date.strftime('%Y-%m-%d')})", 
                              value=f"${close_prices.iloc[-1].item():.2f}")
                # --- Plotting the Trend ---
                st.subheader(f"Historical Trend for {ticker} with Next Day Prediction")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                ax.plot(close_prices.index, close_prices.values, label='Historical Close Price', color='#1f77b4')
                
                # Plot the prediction point
                ax.plot(next_date, prediction, 'ro', markersize=10, label='Predicted Price')
                
                # Annotate the prediction point
                ax.annotate(f"Pred: ${prediction:.2f}", (next_date, prediction), 
                            textcoords="offset points", xytext=(15,-10), 
                            ha='center', fontsize=10, color='red', weight='bold')
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (USD)')
                ax.legend(loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                st.pyplot(fig)

# Display instructions at the bottom
st.markdown("---")
st.markdown("This model predicts the **next day's closing price** based on the previous 60 days of data, including technical indicators (SMA and RSI).")