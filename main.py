import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations, max_loss):
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

            if balance < max_loss:
                break

        ending_balances.append(balance)
        max_drawdowns.append(max(drawdowns))

    return ending_balances, max_drawdowns

# --- Streamlit UI ---
st.title("Futures Trading Monte Carlo Simulator")

# User inputs
risk_percent = st.slider("Risk per Trade (%)", 0.01, 10.0, 0.125, 0.01)
num_trades = st.slider("Number of Trades", 50, 1000, 200, 10)
reward_to_risk = st.slider("Reward-to-Risk Ratio", 1.0, 5.0, 2.0, 0.1)
win_rate = st.slider("Win Rate (%)", 1, 100, 50, 1) / 100
max_loss = st.number_input("Max Loss Stopout ($)", min_value=1000, value=48000, step=1000)

# Run simulation button
if st.button("Run Simulation"):
    initial_balance = 50000
    num_simulations = 5000

    # Run simulation
    ending_balances, max_drawdowns = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations, max_loss
    )

    # Results
    mean_balance = np.mean(ending_balances)
    prob_success = np.sum(np.array(ending_balances) >= 53000) / num_simulations * 100
    prob_bust = np.sum(np.array(ending_balances) < max_loss) / num_simulations * 100
    mean_drawdown = np.mean(max_drawdowns)

    # Display summary
    st.subheader("Simulation Summary")
    st.write(f"Risk per trade: {risk_percent:.3f}%")
    st.write(f"Number of trades: {num_trades}")
    st.write(f"Reward-to-Risk Ratio: {reward_to_risk:.2f}")
    st.write(f"Win Rate: {win_rate:.2%}")
    st.write(f"Average Ending Balance: ${mean_balance:,.2f}")
    st.write(f"Probability of Success (>= $53k): {prob_success:.2f}%")
    st.write(f"Probability of Bust (< ${max_loss:,.0f}): {prob_bust:.2f}%")
    st.write(f"Average Max Drawdown: {mean_drawdown:.2%}")

    # --- Plot histogram with regression ---
    fig, ax = plt.subplots()
    ax.hist(ending_balances, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(53000, color='green', linestyle='dashed', label='Target ($53k)')
    ax.axvline(max_loss, color='red', linestyle='dashed', label=f'Bust (${max_loss:,.0f})')

    # Linear regression
    X = np.array(range(num_simulations)).reshape(-1, 1)
    y = np.array(ending_balances)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    ax.plot(X, y_pred, color='orange', linewidth=2, label='Trend (Linear Regression)')

    ax.set_title(f"Monte Carlo Simulation Results")
    ax.set_xlabel("Simulation #")
    ax.set_ylabel("Ending Balance")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
