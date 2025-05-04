# monte-carlo-risk
 
Professional Futures Monte Carlo Simulator

An interactive Streamlit web app that runs Monte Carlo simulations to model futures trading strategies. It provides advanced risk metrics, equity curve visualizations, drawdown analysis, probability cones, and distribution of outcomes for professional traders.

🚀 Features

Monte Carlo Simulations: Simulate up to 10,000 trading sequences with customizable parameters.

Interactive Inputs:

Risk per trade (% of account)

Win rate (%)

Reward-to-risk ratio (R:R)

Number of trades per simulation

Number of simulations

Max daily loss threshold

Commission/slippage per trade

Risk & Performance Metrics:

Average, maximum, minimum ending balance

Standard deviation of balances

Probability of success (≥ $53,000)

Probability of bust (≤ $0)

Average maximum drawdown

Expectancy (R value)

Profit factor

Average Sharpe ratio (trade-by-trade)

Visualizations:

Distribution of ending balances histogram

Mean equity curve with 10–90% probability cone

Max drawdown distribution chart

Sample cumulative P&L curves (first 50 simulations)

🛠️ Installation

Clone the repo:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

Install dependencies (using Pipenv or venv):

Pipenv:

pipenv install
pipenv shell

venv:

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

Run the app:

streamlit run main.py

📋 Usage

Open the Streamlit UI in your browser (default: http://localhost:8501).

Adjust the sliders and inputs on the sidebar for your strategy parameters.

Click Run Simulation to view metrics and charts.

📖 Code Structure

main.py – Streamlit application entry point.

Key functions:

monte_carlo_simulation(...) – core simulation logic.

calculate_expectancy(...) – computes expectancy.

calculate_profit_factor(...) – estimates profit factor.

calculate_sharpe(...) – calculates Sharpe ratio.

Visualization using matplotlib integrated with Streamlit.

📦 Requirements

Python ≥ 3.8

streamlit

numpy

matplotlib

scikit-learn

Install via:

pip install streamlit numpy matplotlib scikit-learn

🔗 License

MIT License. See LICENSE for details.