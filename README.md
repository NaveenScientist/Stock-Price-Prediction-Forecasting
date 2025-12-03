# Project Title

NIFTY50 Stock Price Forecasting using Deep Learning (LSTM) — Multi-Feature 7-Day Ahead Prediction

# Overview

This project builds a production-ready deep learning forecasting system to predict 7-day ahead stock prices for NIFTY50 stocks.
The model is deployed as a Streamlit analytics dashboard that allows users to track live market prices, view multi-day forecasts, analyze sector-level performance, and discover price trends with interactive visualizations.

# Key Features

Predicts 7-day ahead closing prices using LSTM-based time-series deep learning models

Supports all 50 NIFTY index stocks with automated ticker mapping and sector classification

Live price integration using Yahoo Finance API for real-time comparison against forecasts

Dynamic data updater to fetch missing days and expand historical dataset automatically

Interactive dashboards for market overview, individual stock analysis, and sector performance

Candlestick chart + forecast trajectory for data-centric decision support

⚙️ Technical Architecture
Component	Technology
Data Source	Yahoo Finance (yfinance API)
Historical Data Processing	Pandas + NumPy
ML Model	LSTM (multi-feature OHLCV time-series)
Deployment	Streamlit Web App
Visualization	Plotly (interactive graphs + candlesticks)
Forecast Horizon	7 days ahead for all NIFTY50 stocks

# How the Web App Works

The Streamlit system loads a model summary file containing R² scores and 7-day forecasts for each stock and links it with live prices and sector metadata through dynamic UI components.
Key logic includes:

Sector-wise filter to shortlist stocks

Auto-generated dashboard metrics (best / worst models)

Stock-specific analysis with historical OHLC and forecasted trend graph

Automated data refresh to extend stored CSV files with latest market data from Yahoo Finance 

app

# Results & Business Impact

Achieved 92–95% model accuracy (R²) for majority of NIFTY50 stocks

Enabled risk-aware trading insights by comparing live prices vs forecast expectations

Equipped users with data-driven visual decision support across sectors, stocks, and time windows

Reduced manual market tracking effort through full automation of data refresh and reporting

🏗 Folder Structure
📦 Nifty50 Stock Forecasting
├── app.py  ← Streamlit web application
├── Results_MultiFeature_7day/ ← model summary + forecast results
├── N50 Stock data - *.csv ← historical OHLCV data per stock
└── README.md

# Future Enhancements
Phase	Upcoming Improvement
🔹 Phase 1	Add buy/sell recommendation based on forecast volatility
🔹 Phase 2	Integrate ensemble forecasting (XGBoost + Prophet + LSTM)
🔹 Phase 3	Expand to global indexes (S&P 500, NASDAQ, FTSE, NIKKEI)
🔹 Phase 4	Deploy as cloud API + mobile-responsive dashboard

# Intended Users

Traders & quantitative finance analysts

Portfolio managers

Financial research students

AI/ML enthusiasts interested in time-series forecasting

# 📫 Contact

For collaboration or improvements:

Email: harrinavin@gmail.com

GitHub: github.com/NaveenScientist
