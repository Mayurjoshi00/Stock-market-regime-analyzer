# =====================================
# MARKET REGIME DETECTION + TRADING STRATEGY
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from pandas_datareader import data as pdr

# -------------------------------------
# STEP 1: Load S&P 500 data (STOOQ)
# -------------------------------------
print("Loading S&P 500 data...")

sp500 = pd.read_csv("https://stooq.com/q/d/l/?s=^spx&i=d")
sp500.rename(columns={sp500.columns[0]: "Date"}, inplace=True)
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500.set_index("Date", inplace=True)

print("S&P 500 rows:", len(sp500))

# -------------------------------------
# STEP 2: Load VIX data (FRED)
# -------------------------------------
print("Loading VIX data from FRED...")

vix = pdr.DataReader("VIXCLS", "fred", start=sp500.index.min())
vix.rename(columns={"VIXCLS": "VIX"}, inplace=True)

print("VIX rows:", len(vix))

# -------------------------------------
# STEP 3: Merge datasets
# -------------------------------------
df = sp500[['Close']].join(vix, how="inner")
df.dropna(inplace=True)

print("Merged dataset rows:", len(df))

# -------------------------------------
# STEP 4: Feature Engineering
# -------------------------------------
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['rolling_return'] = df['returns'].rolling(20).mean()
df['rolling_vol'] = df['returns'].rolling(20).std()
df['drawdown'] = (df['Close'] / df['Close'].cummax()) - 1
df['vix_change'] = df['VIX'].pct_change()

df.dropna(inplace=True)

features = df[['returns',
               'rolling_return',
               'rolling_vol',
               'drawdown',
               'VIX',
               'vix_change']]

print("Features created.")

# -------------------------------------
# STEP 5: Train Hidden Markov Model
# -------------------------------------
print("Training HMM model...")

model = GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=1000,
    random_state=42
)

model.fit(features)
df['regime'] = model.predict(features)

print("Model trained.")

# -------------------------------------
# STEP 6: Regime-wise Statistics
# -------------------------------------
print("\n===== REGIME-WISE STATISTICS =====")
print(df.groupby('regime')[['returns', 'rolling_vol']].mean())
print("=================================\n")

# -------------------------------------
# STEP 7: Assign Human-Readable Regime Labels
# -------------------------------------
regime_labels = {
    0: "Bull",
    1: "Strong Bull",
    2: "Bear / Unstable",
    3: "Crisis"
}

df['regime_name'] = df['regime'].map(regime_labels)

# -------------------------------------
# STEP 8: Trading Strategy (Risk-On / Risk-Off)
# -------------------------------------
df['market_return'] = df['returns']
df['strategy_return'] = 0.0

risk_on_regimes = [0, 1]   # Bull & Strong Bull
df.loc[df['regime'].isin(risk_on_regimes), 'strategy_return'] = df['returns']

print("Trading strategy applied.")

# -------------------------------------
# STEP 9: Performance Metrics
# -------------------------------------
def performance_metrics(returns):
    cumulative = (1 + returns).cumprod()
    cagr = cumulative.iloc[-1] ** (252 / len(returns)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = cagr / vol if vol != 0 else 0
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    return cagr, vol, sharpe, max_dd

str_cagr, str_vol, str_sharpe, str_dd = performance_metrics(df['strategy_return'])
mkt_cagr, mkt_vol, mkt_sharpe, mkt_dd = performance_metrics(df['market_return'])

print("----- STRATEGY PERFORMANCE -----")
print(f"CAGR: {str_cagr:.2%}")
print(f"Sharpe Ratio: {str_sharpe:.2f}")
print(f"Max Drawdown: {str_dd:.2%}")

print("\n----- BUY & HOLD PERFORMANCE -----")
print(f"CAGR: {mkt_cagr:.2%}")
print(f"Sharpe Ratio: {mkt_sharpe:.2f}")
print(f"Max Drawdown: {mkt_dd:.2%}")

# -------------------------------------
# STEP 10: Equity Curve Comparison
# -------------------------------------
df['strategy_equity'] = (1 + df['strategy_return']).cumprod()
df['market_equity'] = (1 + df['market_return']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['strategy_equity'], label="Regime Strategy")
plt.plot(df.index, df['market_equity'], label="Buy & Hold", linestyle='--')
plt.legend()
plt.title("Equity Curve: Strategy vs Buy & Hold")
plt.show()

# -------------------------------------
# STEP 11: Market Regime Visualization
# -------------------------------------
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Close'], label="S&P 500", color="black")

for r, name in regime_labels.items():
    plt.scatter(
        df[df['regime'] == r].index,
        df[df['regime'] == r]['Close'],
        label=name,
        s=10
    )

plt.legend()
plt.title("Market Regimes with Labels (HMM)")
plt.show()
