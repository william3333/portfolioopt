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
# Simulate asset returns using Monte Carlo simulation
from pypfopt import monte_carlo


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


    # Number of Monte Carlo simulations
    num_simulations = 10000

    # Simulate returns
    mc_simulated_returns = monte_carlo.portfolio_return(df, num_simulations=num_simulations)
    
    # Convert simulated returns to a DataFrame
    mc_simulated_returns_df = pd.DataFrame(mc_simulated_returns, columns=symbols)
    
    # Calculate the expected returns and covariance matrix from Monte Carlo simulations
    mc_mu = mc_simulated_returns_df.mean()
    mc_S = mc_simulated_returns_df.cov()
    
    # Optimize for max sharpe ratio using Monte Carlo simulated data
    ef = EfficientFrontier(mc_mu, mc_S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    # Display portfolio performance with Monte Carlo simulation
    st.subheader("Portfolio Performance with Monte Carlo Simulation")
    st.write("Optimal Weights:")
    st.write(cleaned_weights)
    ef.portfolio_performance(verbose=True)


    latest_prices = get_latest_prices(df)

    # Input for total portfolio value
    total_portfolio_value = st.number_input("Enter your total portfolio value:", value=15000.0)

    # Calculate discrete allocation
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, _ = da.greedy_portfolio()

    # Display allocation results
    st.subheader("Allocation Results")
    st.write("Discrete allocation:", allocation)
    st.write('Funds remaining: ${:.2f}'.format(leftover))
