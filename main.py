import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# --- Professional Risk Metrics ---
def calculate_expectancy(win_rate, reward_to_risk):
    return (win_rate * reward_to_risk) - (1 - win_rate)

def calculate_profit_factor(ending_balances, risk_amount, reward_amount, win_rate, num_trades):
    wins = win_rate * num_trades
    losses = num_trades - wins
    total_gain = wins * reward_amount
    total_loss = losses * risk_amount
    return total_gain / abs(total_loss) if total_loss != 0 else np.nan

def calculate_sharpe(returns):
    if len(returns) < 2:
        return np.nan
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    return (mean_ret / (std_ret + 1e-9)) * np.sqrt(len(returns))

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk,
                           num_trades, num_simulations, max_daily_loss, commission):
    sims = []
    for _ in range(num_simulations):
        balance = initial_balance
        peak = balance
        daily_loss = 0
        drawdowns = []
        curve = [balance]
        returns = []
        for i in range(num_trades):
            if i % 10 == 0:
                daily_loss = 0
            risk_amt = balance * (risk_percent / 100)
            reward_amt = risk_amt * reward_to_risk
            pnl = reward_amt if np.random.rand() < win_rate else -risk_amt
            if pnl < 0:
                daily_loss += abs(pnl)
            balance += pnl - commission
            returns.append(pnl)
            curve.append(balance)
            peak = max(peak, balance)
            drawdowns.append((peak - balance) / peak)
            if daily_loss > max_daily_loss or balance < 0:
                break
        sims.append({
            'curve': np.array(curve),
            'ending': balance,
            'max_dd': np.max(drawdowns) if drawdowns else 0,
            'returns': np.array(returns)
        })
    return sims

# --- Streamlit App ---
def main():
    st.title("Professional Futures Monte Carlo Simulator")

    # Sidebar Inputs
    starting_capital = st.number_input(
        "Starting Capital ($)",
        min_value=1000.0,
        max_value=1_000_000.0,
        value=50_000.0,
        step=1_000.0
    )
    risk_percent   = st.slider("Risk per Trade (%)", 0.01, 10.0, 0.125, 0.01)
    win_rate       = st.slider("Win Rate (%)", 1, 99, 50, 1) / 100
    rr             = st.slider("Reward-to-Risk Ratio", 1.0, 5.0, 2.0, 0.1)
    trades         = st.slider("Trades per Simulation", 50, 1000, 200, 10)
    sims_count     = st.slider("Simulations", 100, 10000, 5000, 100)
    max_daily_loss = st.number_input(
        "Max Daily Loss ($)",
        min_value=0.0,
        max_value=starting_capital,
        value=1000.0,
        step=100.0
    )
    bust_threshold = st.number_input(
        "Bust Threshold ($)",
        min_value=0.0,
        max_value=starting_capital,
        value=0.0,
        step=100.0
    )
    commission     = st.number_input("Commission per Trade ($)", 0.0, 50.0, 2.0, 0.5)
    target_balance = st.number_input(
        "Success Target ($)",
        min_value=starting_capital,
        max_value=starting_capital * 2,
        value=starting_capital + 3_000.0,
        step=100.0
    )

    if st.button("Run Simulation"):
        sims = monte_carlo_simulation(
            starting_capital, risk_percent, win_rate, rr,
            trades, sims_count, max_daily_loss, commission
        )
        endings    = np.array([s['ending'] for s in sims])
        max_dds    = np.array([s['max_dd']  for s in sims])
        sharpe_vals = np.array([calculate_sharpe(s['returns']) for s in sims])

        # Risk metrics
        expectancy = calculate_expectancy(win_rate, rr)
        pfactor    = calculate_profit_factor(
            endings,
            starting_capital * (risk_percent / 100),
            starting_capital * (risk_percent / 100) * rr,
            win_rate, trades
        )

        # Summary
        st.subheader("Summary Metrics")
        st.write(f"Starting Capital:         ${starting_capital:,.2f}")
        st.write(f"Average Ending Balance:   ${np.mean(endings):,.2f}")
        st.write(f"Max Ending Balance:       ${np.max(endings):,.2f}")
        st.write(f"Min Ending Balance:       ${np.min(endings):,.2f}")
        st.write(f"Std Dev of Balances:      ${np.std(endings):,.2f}")
        st.write(f"Success Rate (≥ ${target_balance:,.0f}): {np.mean(endings >= target_balance)*100:.2f}%")
        st.write(f"Bust Rate   (< ${bust_threshold:,.0f}): {np.mean(endings  < bust_threshold)*100:.2f}%")
        st.write(f"Avg Max Drawdown:         {np.mean(max_dds):.2%}")
        st.write(f"Expectancy (R):            {expectancy:.2f}")
        st.write(f"Profit Factor:            {pfactor:.2f}")
        st.write(f"Avg Sharpe Ratio:         {np.nanmean(sharpe_vals):.2f}")

        # Outcome Distribution
        fig1, ax1 = plt.subplots(figsize=(10,6))
        ax1.hist(endings, bins=50, color='skyblue', edgecolor='black')
        ax1.axvline(target_balance, color='green', linestyle='--', label='Target')
        ax1.axvline(bust_threshold, color='red', linestyle='--', label='Bust')
        ax1.set(title='Outcome Distribution', xlabel='Ending Balance', ylabel='Frequency')
        ax1.legend()
        st.pyplot(fig1)

        # Equity Curve + Probability Cone
        curves  = np.array([s['curve'] for s in sims])
        max_len = max(len(c) for c in curves)
        padded  = np.array([np.pad(c, (0, max_len-len(c)), 'edge') for c in curves])
        mean_c  = np.mean(padded, axis=0)
        p10      = np.percentile(padded, 10, axis=0)
        p90      = np.percentile(padded, 90, axis=0)

        fig2, ax2 = plt.subplots(figsize=(10,6))
        x = np.arange(max_len)
        ax2.plot(x, mean_c, color='black', label='Mean Equity')
        ax2.fill_between(x, p10, p90, color='gray', alpha=0.3, label='10-90% Cone')
        ax2.set(title='Equity Curve w/ Probability Cone', xlabel='Trade #', ylabel='Balance')
        ax2.legend()
        st.pyplot(fig2)

        # Max Drawdown Distribution
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.hist(max_dds, bins=50, color='salmon', edgecolor='black')
        ax3.set(title='Max Drawdown Distribution', xlabel='Max Drawdown', ylabel='Frequency')
        st.pyplot(fig3)

        # Sample Cumulative PnL Curves
        fig4, ax4 = plt.subplots(figsize=(10,6))
        for s in sims[:50]:
            ax4.plot(np.cumsum(s['returns']), alpha=0.3)
        ax4.set(title='Sample Cumulative PnL Curves', xlabel='Trade #', ylabel='Cumulative PnL')
        st.pyplot(fig4)

if __name__ == '__main__':
    main()
