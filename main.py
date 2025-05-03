import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# --- Professional Risk Metrics ---
def calculate_expectancy(win_rate, reward_to_risk):
    return (win_rate * reward_to_risk) - (1 - win_rate)

def calculate_profit_factor(ending_balances, risk_amount, reward_amount, win_rate, num_trades):
    # Approximates profit factor: total wins / total losses
    wins = win_rate * num_trades
    losses = num_trades - wins
    total_gain = wins * reward_amount
    total_loss = losses * risk_amount
    return total_gain / abs(total_loss) if total_loss != 0 else np.nan

def calculate_sharpe(returns):
    # Sharpe ratio for trade-by-trade returns (risk-free rate=0)
    return np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(len(returns))

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk,
                           num_trades, num_simulations, max_daily_loss, commission):
    results = []
    for _ in range(num_simulations):
        balance = initial_balance
        peak = balance
        daily_loss = 0
        drawdowns = []
        returns = []
        for i in range(num_trades):
            if i % 10 == 0:
                daily_loss = 0
            risk_amount = balance * (risk_percent / 100)
            reward_amount = risk_amount * reward_to_risk
            if np.random.rand() < win_rate:
                pnl = reward_amount
            else:
                pnl = -risk_amount
                daily_loss += risk_amount
            balance += pnl - commission
            returns.append(pnl)
            peak = max(peak, balance)
            drawdowns.append((peak - balance) / peak)
            if daily_loss > max_daily_loss or balance < 0:
                break
        results.append({
            'ending_balance': balance,
            'max_drawdown': max(drawdowns) if drawdowns else 0,
            'returns': returns
        })
    return results

# --- Streamlit UI ---
st.title("Professional Futures Trading Monte Carlo Simulator")
# Inputs
risk_percent        = st.slider("Risk per Trade (%)", 0.01, 10.0, 0.125, 0.01)
win_rate            = st.slider("Win Rate (%)", 1, 99, 50, 1) / 100
reward_to_risk      = st.slider("Reward-to-Risk Ratio", 1.0, 5.0, 2.0, 0.1)
num_trades          = st.slider("Number of Trades/Sim", 50, 1000, 200, 10)
num_simulations     = st.slider("Number of Simulations", 100, 10000, 5000, 100)
max_daily_loss      = st.number_input("Max Daily Loss ($)", 100.0, 5000.0, 1000.0, 50.0)
commission          = st.number_input("Commission per Trade ($)", 0.0, 50.0, 2.0, 0.5)

if st.button("Run Simulation"):
    initial_balance = 50000
    data = monte_carlo_simulation(initial_balance, risk_percent, win_rate,
                                  reward_to_risk, num_trades, num_simulations,
                                  max_daily_loss, commission)
    ending_balances = np.array([d['ending_balance'] for d in data])
    max_drawdowns   = np.array([d['max_drawdown'] for d in data])
    all_returns     = np.hstack([d['returns'] for d in data])

    # Risk Metrics
    expectancy     = calculate_expectancy(win_rate, reward_to_risk)
    avg_return     = np.mean(all_returns)
    sharpe_ratio   = calculate_sharpe(all_returns)
    avg_drawdown   = np.mean(max_drawdowns)
    profit_factor  = calculate_profit_factor(ending_balances, initial_balance * (risk_percent/100),
                                             initial_balance * (risk_percent/100)*reward_to_risk,
                                             win_rate, num_trades)

    # Summary
    st.subheader("Simulation Summary & Risk Metrics")
    st.write(f"Avg Ending Balance: ${np.mean(ending_balances):,.2f}")
    st.write(f"Probability of Success (>= $53k): {np.mean(ending_balances>=53000)*100:.2f}%")
    st.write(f"Probability of Bust (< $0): {np.mean(ending_balances<=0)*100:.2f}%")
    st.write(f"Avg Max Drawdown: {avg_drawdown:.2%}")
    st.write(f"Expectancy per Trade: {expectancy:.2f} R")
    st.write(f"Profit Factor: {profit_factor:.2f}")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Linear Regression on Ending Balances vs Simulation #
    X = np.arange(len(ending_balances)).reshape(-1,1)
    y = ending_balances
    lr = LinearRegression().fit(X,y)
    trend = lr.predict(X)

    # Histogram + Regression
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.hist(ending_balances, bins=50, color='skyblue', edgecolor='black')
    ax1.plot([], [])  # placeholder
    ax1.plot(np.linspace(0,len(ending_balances),100),
             np.interp(np.linspace(0,len(ending_balances),100), X.flatten(), trend),
             color='orange', label='Linear Trend')
    ax1.axvline(53000, color='green', linestyle='--', label='Target')
    ax1.set(title='Ending Balance Distribution & Trend', xlabel='Ending Balance', ylabel='Frequency')
    ax1.legend()
    st.pyplot(fig1)

    # Sample PnL Curves
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for d in data[:50]:
        ax2.plot(d['returns'], alpha=0.3)
    ax2.set(title='Sample PnL Return Curves (first 50 sims)', xlabel='Trade #', ylabel='PnL ($)')
    st.pyplot(fig2)
