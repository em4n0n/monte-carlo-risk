import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations):
    ending_balances = []
    max_drawdowns = []
    pnl_curves = []

    for _ in range(num_simulations):
        balance = initial_balance
        peak = balance
        drawdowns = []
        pnl_curve = [balance]

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
            pnl_curve.append(balance)

            if balance < 48000:
                break

        ending_balances.append(balance)
        max_drawdowns.append(max(drawdowns))
        pnl_curves.append(pnl_curve)

    return ending_balances, max_drawdowns, pnl_curves

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
    ending_balances, max_drawdowns, pnl_curves = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations
    )

    # Results
    mean_balance = np.mean(ending_balances)
    median_balance = np.median(ending_balances)
    std_balance = np.std(ending_balances)
    prob_success = np.sum(np.array(ending_balances) >= 53000) / num_simulations * 100
    prob_bust = np.sum(np.array(ending_balances) < 48000) / num_simulations * 100
    mean_drawdown = np.mean(max_drawdowns)
    max_drawdown = np.max(max_drawdowns)
    min_balance = np.min(ending_balances)
    max_balance = np.max(ending_balances)

    # Displaying results
    st.subheader("Simulation Summary")
    st.write(f"Risk per trade: {risk_percent:.3f}%")
    st.write(f"Number of trades: {num_trades}")
    st.write(f"Reward-to-Risk Ratio: {reward_to_risk:.2f}")
    st.write(f"Average Ending Balance: ${mean_balance:,.2f}")
    st.write(f"Median Ending Balance: ${median_balance:,.2f}")
    st.write(f"Standard Deviation of Ending Balances: ${std_balance:,.2f}")
    st.write(f"Min Ending Balance: ${min_balance:,.2f}")
    st.write(f"Max Ending Balance: ${max_balance:,.2f}")
    st.write(f"Probability of Success (>= $53k): {prob_success:.2f}%")
    st.write(f"Probability of Bust (< $48k): {prob_bust:.2f}%")
    st.write(f"Average Max Drawdown: {mean_drawdown:.2%}")
    st.write(f"Max Observed Drawdown: {max_drawdown:.2%}")

    # Plot ending balance distribution
    fig1, ax1 = plt.subplots()
    ax1.hist(ending_balances, bins=50, color='skyblue', edgecolor='black')
    ax1.axvline(53000, color='green', linestyle='dashed', label='Target ($53k)')
    ax1.axvline(48000, color='red', linestyle='dashed', label='Bust ($48k)')
    ax1.set_title(f"Monte Carlo Simulation ({risk_percent}% Risk)")
    ax1.set_xlabel("Ending Balance")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # Plot PnL Curves
    fig2, ax2 = plt.subplots()
    for i in range(min(50, len(pnl_curves))):
        ax2.plot(pnl_curves[i], alpha=0.3)
    ax2.set_title("Sample PnL Curves from Simulations")
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Balance")
    ax2.grid(True)
    st.pyplot(fig2)
