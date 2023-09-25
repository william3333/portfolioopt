import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Set the page title
st.title("Portfolio Optimization with Streamlit")

# Input for ticker symbols
symbols_input = st.text_input("Enter ticker symbols separated by commas (e.g., AAPL, GOOG, MSFT):")

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

    # Optimize for max sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Display portfolio performance
    st.subheader("Portfolio Performance")
    st.write("Optimal Weights:")
    st.write(cleaned_weights)
    ef.portfolio_performance(verbose=True)

    latest_prices = get_latest_prices(df)

    # Input for total portfolio value
    total_portfolio_value = st.number_input("Enter your total portfolio value:", value=15000.0, step=1000.0)

    # Calculate discrete allocation
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()

    # Display allocation results
    st.subheader("Allocation Results")
    st.write("Discrete allocation:", allocation)
    st.write('Funds remaining: ${:.2f}'.format(leftover))
