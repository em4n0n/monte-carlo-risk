import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations):
    ending_balances = []
    max_drawdowns = []

    for _ in range(num_simulations):
        balance = initial_balance
        peak = balance
        drawdowns = []

        for _ in range(num_trades):
            risk_amount = balance * (risk_percent / 100)
            reward_amount = risk_amount * reward_to_risk

            if np.random.rand() < win_rate:
                balance += reward_amount
            else:
                balance -= risk_amount

            peak = max(peak, balance)
            drawdown = (peak - balance) / peak
            drawdowns.append(drawdown)

            if balance < 48000:
                break

        ending_balances.append(balance)
        max_drawdowns.append(max(drawdowns))

    return ending_balances, max_drawdowns


# --- Streamlit UI ---
st.title("Futures Trading Monte Carlo Simulator")

# Risk per trade scale
risk_percent = st.slider("Risk per Trade (%)", 0.01, 2.0, 0.125, 0.01)

# Number of trades
num_trades = st.slider("Number of Trades", 50, 1000, 200, 10)

# Reward-to-Risk Ratio scale
reward_to_risk = st.slider("Reward-to-Risk Ratio", 1.0, 5.0, 2.0, 0.1)

# Run simulation button
if st.button("Run Simulation"):
    # Fixed parameters
    initial_balance = 50000
    win_rate = 0.5
    num_simulations = 5000

    # Run simulation
    ending_balances, max_drawdowns = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations
    )

    # Results
    mean_balance = np.mean(ending_balances)
    prob_success = np.sum(np.array(ending_balances) >= 53000) / num_simulations * 100
    prob_bust = np.sum(np.array(ending_balances) < 48000) / num_simulations * 100
    mean_drawdown = np.mean(max_drawdowns)

    # Displaying results
    st.subheader("Simulation Summary")
    st.write(f"Risk per trade: {risk_percent:.3f}%")
    st.write(f"Number of trades: {num_trades}")
    st.write(f"Reward-to-Risk Ratio: {reward_to_risk:.2f}")
