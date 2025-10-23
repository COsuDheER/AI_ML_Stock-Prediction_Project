LSTM Stock Price Trend Prediction

Project Overview

This project implements a complete Machine Learning Engineering pipeline to forecast the next-day closing price trend of a stock using a Long Short-Term Memory (LSTM) neural network. The analysis is performed on highly volatile stock data (e.g., TSLA or AAPL) and features an end-to-end framework including advanced feature engineering and a production-ready web application dashboard.

Key Objectives Achieved:

Data acquisition, cleaning, and normalization.

Integration of financial indicators (SMA and RSI) as input features.

Design and training of a multi-layer LSTM model.

Deployment of an interactive web application using Streamlit.

🛠️ Tools and Libraries

|

| Category | Tool/Library | Purpose |
| Data Handling | Pandas, NumPy | Data manipulation and numerical operations. |
| Data Source | yfinance | Fetching real-time and historical stock data. |
| Modeling | TensorFlow, Keras | Building and training the LSTM deep learning model. |
| Preprocessing | scikit-learn | Data scaling (MinMaxScaler) and metric calculation. |
| Deployment | Streamlit | Creating the interactive, user-facing prediction dashboard. |
| Serialization | joblib | Saving and loading the trained model and scaler object. |

🚀 Getting Started

To run this project, you need a Python environment (preferably 3.8+) and the necessary packages.

1. Prerequisites

First, install all required dependencies:

pip install tensorflow pandas numpy yfinance scikit-learn streamlit joblib matplotlib



2. File Structure

Ensure the following files are present in your project root directory:

| File Name | Description |
| ai_ml_project_stock_prediction.ipynb | The main Jupyter Notebook containing all training, evaluation, and analysis steps. |
| app.py | The standalone Python script used to run the Streamlit dashboard. |
| lstm_model_multi.keras | The trained LSTM model weights (saved after training with indicators). |
| scaler_multi.pkl | The fitted MinMaxScaler object used to normalize the multi-feature input data. |

⚙️ Project Pipeline: Steps Involved

The project follows a structured data science workflow, implemented primarily in the Jupyter Notebook (.ipynb).

Phase 1: Data Preparation and Feature Engineering

| Step | Description | Implementation Details |
| Data Fetching | Used yfinance to retrieve several years of historical stock data (e.g., TSLA) up to a recent date (e.g., Oct 2025). | Fetched the Close price along with High, Low, and Volume. |
| Feature Creation | Calculated technical indicators to enhance model input. | Included 20-Day Simple Moving Average (SMA) and 14-Day Relative Strength Index (RSI). |
| Normalization | Applied MinMaxScaler to scale all features to the range $[0, 1]$, which is crucial for LSTM stability. | The scaler was saved (scaler_multi.pkl) for inverse transformation during prediction. |
| Sequence Formatting | Transformed the time series data into supervised sequences. | Input (X): 180 previous days of multi-feature data. Target (y): The next day's normalized closing price. |
| Split | Data was split chronologically (80% Train, 20% Test). | No shuffling was performed to maintain temporal integrity. |

Phase 2: Model Building and Training

| Step | Description | Implementation Details |
| Architecture | Built a Sequential Keras model tailored for sequence prediction. | Two stacked LSTM layers (50 units each) with 20% Dropout, followed by a final Dense layer (1 unit). |
| Compilation | Compiled the model for regression. | Used Adam optimizer and mean_squared_error (MSE) loss function. |
| Training | Trained the multi-feature model. | Used 25 epochs with a batch size of 32. |
| Serialization | Saved the final trained model and scaler. | model_multi.save('lstm_model_multi.keras') and joblib.dump(scaler_multi, 'scaler_multi.pkl'). |

Phase 3: Deployment (Streamlit Dashboard)

The final model is deployed via the app.py script.

How to Run the Dashboard

Open your Command Prompt or Terminal.

Navigate to the directory containing app.py, lstm_model_multi.keras, and scaler_multi.pkl.

Run the following command:

streamlit run app.py



The application will automatically open in your web browser, allowing you to enter a stock ticker and generate a next-day price prediction instantly.

📊 Final Results and Analysis

The final multi-feature model achieved improved performance over the single-feature baseline. After retraining the model on data up to October 2025, the model predicted a significant price reversal:

Last Recorded Price: $438.97

Predicted Next Price: $339.56

Analysis: The large price difference ($\approx \$99$) suggests the LSTM model interprets the recent steep price increase as highly volatile and unsustainable, predicting a strong mean reversion toward the recent long-term average. This highlights the model's sensitivity to sudden, sharp trend changes in highly non-stationary financial data.

The application will automatically open in your web browser, allowing you to enter a stock ticker and generate a next-day price prediction instantly.📊 Final Results and AnalysisThe final multi-feature model achieved improved performance over the single-feature baseline. After retraining the model on data up to October 2025, the model predicted a significant price reversal:Last Recorded Price: $438.97Predicted Next Price: $339.56Analysis: The large price difference ($\approx \$99$) suggests the LSTM model interprets the recent steep price increase as highly volatile and unsustainable, predicting a strong mean reversion toward the recent long-term average. This highlights the model's sensitivity to sudden, sharp trend changes in highly non-stationary financial data.
