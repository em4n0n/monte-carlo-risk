import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

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
risk_percent = st.slider("Risk per Trade (%)", 0.01, 10.0, 0.125, 0.01)

# Win rate
win_rate = st.slider("Win Rate (%)", 1, 99, 50, 1) / 100

# Max daily loss
max_daily_loss = st.number_input("Max Daily Loss ($)", min_value=100.0, max_value=5000.0, value=1000.0, step=50.0)

# Number of trades
num_trades = st.slider("Number of Trades", 50, 1000, 200, 10)

# Reward-to-Risk Ratio scale
reward_to_risk = st.slider("Reward-to-Risk Ratio", 1.0, 5.0, 2.0, 0.1)

# Number of simulations
num_simulations = st.slider("Number of Simulations", 100, 10000, 5000, 100)

# Run simulation button
if st.button("Run Simulation"):
    # Fixed parameters
    initial_balance = 50000

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
    st.write(f"Win rate: {win_rate * 100:.1f}%")
    st.write(f"Number of trades: {num_trades}")
    st.write(f"Reward-to-Risk Ratio: {reward_to_risk:.2f}")
    st.write(f"Max Daily Loss: ${max_daily_loss:,.2f}")
    st.write(f"Average Ending Balance: ${mean_balance:,.2f}")
    st.write(f"Probability of Success (>= $53k): {prob_success:.2f}%")
    st.write(f"Probability of Bust (< $48k): {prob_bust:.2f}%")
    st.write(f"Average Max Drawdown: {mean_drawdown:.2%}")

    # Linear Regression on Results
    X = np.array([[risk_percent, win_rate, reward_to_risk]])  # Input features
    y = np.array(ending_balances)  # Target: ending balances

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Display linear regression results
    st.subheader("Linear Regression Results")
    st.write(f"Intercept: {model.intercept_:.2f}")
    st.write(f"Coefficients: Risk %: {model.coef_[0]:.2f}, Win Rate: {model.coef_[1]:.2f}, Reward-to-Risk: {model.coef_[2]:.2f}")

    # Plotting linear regression line
    fig_lr, ax_lr = plt.subplots()
    ax_lr.scatter([risk_percent], y, color='blue', label="Data Points")
    ax_lr.plot([risk_percent], model.predict(X), color='red', label="Linear Regression Line")
    ax_lr.set_title("Linear Regression: Ending Balance vs. Risk Factors")
    ax_lr.set_xlabel("Risk Percent")
    ax_lr.set_ylabel("Ending Balance")
    ax_lr.legend()
    st.pyplot(fig_lr)

    # Plot results
    fig, ax = plt.subplots()
    ax.hist(ending_balances, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(53000, color='green', linestyle='dashed', label='Target ($53k)')
    ax.axvline(48000, color='red', linestyle='dashed', label='Bust ($48k)')
    ax.set_title(f"Monte Carlo Simulation ({risk_percent}% Risk)")
    ax.set_xlabel("Ending Balance")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
