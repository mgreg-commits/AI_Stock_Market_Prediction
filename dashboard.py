
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO
from trading_env import MultiStockTradingEnv

# Load and prepare data
df = pd.read_csv("processed_stock_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['Ticker', 'date'])

# Load model
model = PPO.load("ppo_aapl_trading_agent")

# Tabs for single-ticker and portfolio view
tab1, tab2 = st.tabs(["📊 Single Ticker View", "💼 Portfolio View"])

# ---------------------- Tab 1: Single Ticker ---------------------- #
with tab1:
    tickers = df['Ticker'].unique()
    selected_ticker = st.sidebar.selectbox("Select a Stock Ticker", tickers)

    ticker_df = df[df['Ticker'] == selected_ticker].reset_index(drop=True)
    min_date, max_date = ticker_df['date'].min(), ticker_df['date'].max()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    # Set up environment
    env = MultiStockTradingEnv(df, ticker=selected_ticker)
    obs = env.reset()

    net_worths, actions, prices, dates = [], [], [], []
    rsi_vals, macd_vals, macd_signal_vals = [], [], []
    trade_log = []

    done = False
    while not done:
        if np.isnan(obs).any():
            break
        action, _ = model.predict(obs.reshape(1, -1))
        obs, reward, done, _ = env.step(action)
        row = env.df.loc[env.current_step]
        if not (start_date <= row['date'].date() <= end_date):
            continue
        date = row['date']
        price = row['Close']
        net_worths.append(env.net_worth)
        actions.append(int(action))
        prices.append(price)
        dates.append(date)
        rsi_vals.append(row['RSI_14'])
        macd_vals.append(row['MACD'])
        macd_signal_vals.append(row['MACD_signal'])
        if int(action) in [1, 2]:
            trade_log.append({
                "Date": date,
                "Action": "Buy" if int(action) == 1 else "Sell",
                "Price": price,
                "Net Worth": env.net_worth
            })

    trade_log_df = pd.DataFrame(trade_log)
    buy_points = [i for i, a in enumerate(actions) if a == 1]
    sell_points = [i for i, a in enumerate(actions) if a == 2]

    initial_price = prices[0] if prices else 1
    initial_net = net_worths[0] if net_worths else 1
    rl_return = [(nw - initial_net) / initial_net * 100 for nw in net_worths]
    buy_hold_return = [(p - initial_price) / initial_price * 100 for p in prices]
    returns = pd.Series(net_worths).pct_change().fillna(0)
    rolling_vol = returns.rolling(window=10).std() * 100

    st.title(f"RL Trading Dashboard — {selected_ticker}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
    fig_price.add_trace(go.Scatter(x=[dates[i] for i in buy_points], y=[prices[i] for i in buy_points], mode='markers',
                                   marker=dict(color='green', symbol='triangle-up', size=10), name='Buy'))
    fig_price.add_trace(go.Scatter(x=[dates[i] for i in sell_points], y=[prices[i] for i in sell_points], mode='markers',
                                   marker=dict(color='red', symbol='triangle-down', size=10), name='Sell'))
    fig_price.add_trace(go.Scatter(x=dates, y=[initial_price + (p - initial_price) for p in prices],
                                   mode='lines', name='Buy & Hold Benchmark', line=dict(dash='dash', color='gray')))
    fig_price.update_layout(title="Price + Trades + Benchmark", xaxis_title="Date", yaxis_title="Price", xaxis=dict(type='date'), height=500)
    st.plotly_chart(fig_price, use_container_width=True)

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=dates, y=net_worths, mode='lines', name='Net Worth'))
    fig_net.update_layout(title='Agent Net Worth', xaxis_title='Date', yaxis_title='USD', xaxis=dict(type='date'))
    st.plotly_chart(fig_net, use_container_width=True)

    fig_return = go.Figure()
    fig_return.add_trace(go.Scatter(x=dates, y=rl_return, name='RL Strategy'))
    fig_return.add_trace(go.Scatter(x=dates, y=buy_hold_return, name='Buy & Hold'))
    fig_return.update_layout(title="Cumulative Return (%)", xaxis_title="Date", yaxis_title="Return (%)", xaxis=dict(type='date'))
    st.plotly_chart(fig_return, use_container_width=True)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=dates, y=rolling_vol, name='Volatility (%)', line=dict(color='red')))
    fig_vol.update_layout(title='Agent Volatility (10-Day Rolling)', xaxis_title='Date', yaxis_title='Volatility (%)', xaxis=dict(type='date'))
    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Trade Log")
    if not trade_log_df.empty:
        st.dataframe(trade_log_df)
        csv = trade_log_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Trade Log as CSV", csv, file_name=f"{selected_ticker}_trades.csv")
    else:
        st.info("No trades executed.")

# ---------------------- Tab 2: Portfolio View ---------------------- #
with tab2:
    st.header("RL Portfolio Simulation — Top Tickers")
    selected = st.multiselect("Select tickers for portfolio", df['Ticker'].unique(), default=list(df['Ticker'].unique())[:5])

    portfolio_df = pd.DataFrame()
    for ticker in selected:
        env = MultiStockTradingEnv(df, ticker=ticker)
        obs = env.reset()
        ticker_data = []
        done = False
        while not done:
            if np.isnan(obs).any():
                break
            action, _ = model.predict(obs.reshape(1, -1))
            obs, reward, done, _ = env.step(action)
            row = env.df.loc[env.current_step]
            ticker_data.append({'Date': row['date'], 'Net Worth': env.net_worth, 'Ticker': ticker})
        portfolio_df = pd.concat([portfolio_df, pd.DataFrame(ticker_data)], ignore_index=True)

    if not portfolio_df.empty:
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        pivot_df = portfolio_df.pivot(index='Date', columns='Ticker', values='Net Worth').fillna(method='ffill')
        pivot_df['Total'] = pivot_df.sum(axis=1)
        returns = pivot_df['Total'].pct_change().fillna(0)
        vol = returns.rolling(10).std() * 100
        cumulative_return = (pivot_df['Total'] - pivot_df['Total'].iloc[0]) / pivot_df['Total'].iloc[0] * 100

        fig = go.Figure()
        for col in pivot_df.columns[:-1]:
            fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df[col], name=col))
        fig.update_layout(title='Net Worth per Ticker', xaxis_title='Date', yaxis_title='Net Worth', height=500)
        st.plotly_chart(fig, use_container_width=True)

        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df['Total'], name='Portfolio'))
        fig_total.update_layout(title='Total Portfolio Value', xaxis_title='Date', yaxis_title='USD')
        st.plotly_chart(fig_total, use_container_width=True)

        fig_return = go.Figure()
        fig_return.add_trace(go.Scatter(x=pivot_df.index, y=cumulative_return, name='Portfolio Return'))
        fig_return.update_layout(title='Cumulative Portfolio Return (%)', xaxis_title='Date', yaxis_title='Return (%)')
        st.plotly_chart(fig_return, use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=pivot_df.index, y=vol, name='Volatility (%)', line=dict(color='red')))
        fig_vol.update_layout(title='Portfolio Volatility (10-Day Rolling)', xaxis_title='Date', yaxis_title='Volatility (%)')
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning("No portfolio data available.")
