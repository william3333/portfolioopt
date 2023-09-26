from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)

import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pandas_datareader import data as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Set the page title
st.title("Portfolio Optimization")

# Input for ticker symbols
symbols_input = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL, GOOG, MSFT):", value="AAPL, GOOG, MSFT")

# Check if symbols are provided
if symbols_input:
    symbols = [s.strip() for s in symbols_input.split(',')]

    # Define the start and end dates
    start_date = "2013-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Create an empty DataFrame to store the data
    df = pd.DataFrame()

    # Fetch the data for each symbol and add it to the DataFrame
    for symbol in symbols:
        data = yf.download(symbol, start=start_date, end=end_date)
        df[symbol] = data["Adj Close"]

    returns = df.pct_change()
    cov_matrix_annual = returns.cov() * 252

    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Perform Monte Carlo simulation for portfolio optimization
    num_portfolios = 10000
    results = np.zeros((4, num_portfolios))
    risk_free_rate = 0.02

    for i in range(num_portfolios):
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mu * weights) * 252
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_stddev
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_stddev
        results[3,i] = weights.sum()

    # Get the optimal portfolio from Monte Carlo simulation
    max_sharpe_idx = results[2].argmax()
    optimal_weights = results[3, max_sharpe_idx]

    # Display portfolio performance
    st.subheader("Portfolio Performance")
    st.write("Optimal Weights (Monte Carlo):")
    st.write(optimal_weights)
    st.write("Expected Portfolio Performance (Monte Carlo):")
    st.write("Expected Return:", results[0, max_sharpe_idx])
    st.write("Expected Risk (Standard Deviation):", results[1, max_sharpe_idx])

    latest_prices = get_latest_prices(df)

    # Input for total portfolio value
    total_portfolio_value = st.number_input("Enter your total portfolio value:", value=15000.0)

    # Calculate discrete allocation
    allocation = {symbol: optimal_weights for symbol in symbols}
    da = DiscreteAllocation(allocation, latest_prices, total_portfolio_value=total_portfolio_value)
    discrete_allocation, _ = da.greedy_portfolio()

    # Display allocation results
    st.subheader("Allocation Results")
    st.write("Discrete allocation:", discrete_allocation)
