import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations, max_daily_loss, commission_per_trade, strategy_type, compounding):
    ending_balances = []
    max_drawdowns = []
    pnl_curves = []

    for _ in range(num_simulations):
        balance = initial_balance
        peak = balance
        drawdowns = []
        pnl_curve = [balance]
        daily_loss = 0
        current_risk_percent = risk_percent

        for i in range(num_trades):
            if i % 10 == 0:
                daily_loss = 0  # Reset daily loss every 10 trades (approx 1 day)

            risk_amount = balance * (current_risk_percent / 100) if compounding else initial_balance * (current_risk_percent / 100)
            reward_amount = risk_amount * reward_to_risk

            if np.random.rand() < win_rate:
                balance += reward_amount
                current_risk_percent = min(current_risk_percent * 2, 10.0)
            else:
                balance -= risk_amount
                current_risk_percent = max(current_risk_percent / 2, 0.0005)
                daily_loss += risk_amount

            balance -= commission_per_trade  # Deduct commission

            if daily_loss > max_daily_loss:
                break

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
risk_percent = st.slider("Initial Risk per Trade (%)", 0.0005, 10.0, 0.125, 0.0005)

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

# Commission per trade
commission_per_trade = st.number_input("Commission/Slippage per Trade ($)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

# Strategy profile
strategy_type = st.selectbox("Strategy Type", ["Scalping", "Day Trading", "Swing Trading"])

# Compounding mode
compounding = st.checkbox("Use Compounding Risk", value=True)

# PnL Curve Filter
pnl_filter = st.radio("Show PnL Curves for:", ("All Runs", "Only Successful", "Only Bust"))

# Run simulation button
if st.button("Run Simulation"):
    # Fixed parameters
    initial_balance = 50000

    # Run simulation
    ending_balances, max_drawdowns, pnl_curves = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, reward_to_risk, num_trades, num_simulations, max_daily_loss,
        commission_per_trade, strategy_type, compounding
    )

    # Convert results to arrays
    ending_balances = np.array(ending_balances)
    max_drawdowns = np.array(max_drawdowns)

    # Compute metrics
    mean_balance = np.mean(ending_balances)
    median_balance = np.median(ending_balances)
    std_balance = np.std(ending_balances)
    prob_success = np.sum(ending_balances >= 53000) / num_simulations * 100
    prob_bust = np.sum(ending_balances < 48000) / num_simulations * 100
    mean_drawdown = np.mean(max_drawdowns)
    max_drawdown = np.max(max_drawdowns)
    min_balance = np.min(ending_balances)
    max_balance = np.max(ending_balances)

    # Displaying results
    st.subheader("Simulation Summary")
    st.write(f"Risk per trade: {risk_percent:.3f}%")
    st.write(f"Win rate: {win_rate * 100:.1f}%")
    st.write(f"Number of trades: {num_trades}")
    st.write(f"Reward-to-Risk Ratio: {reward_to_risk:.2f}")
    st.write(f"Max Daily Loss: ${max_daily_loss:,.2f}")
    st.write(f"Commission per Trade: ${commission_per_trade:.2f}")
    st.write(f"Strategy Type: {strategy_type}")
    st.write(f"Compounding Risk: {'Yes' if compounding else 'No'}")
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

    # Filter PnL curves
    if pnl_filter == "Only Successful":
        indices = np.where(ending_balances >= 53000)[0]
    elif pnl_filter == "Only Bust":
        indices = np.where(ending_balances < 48000)[0]
    else:
        indices = np.arange(len(pnl_curves))

    # Plot filtered PnL Curves
    fig2, ax2 = plt.subplots()
    for i in indices[:50]:
        ax2.plot(pnl_curves[i], alpha=0.3)
    ax2.set_title(f"Sample PnL Curves - {pnl_filter}")
    ax2.set_xlabel("Trade Number")
    ax2.set_ylabel("Balance")
    ax2.grid(True)
    st.pyplot(fig2)
