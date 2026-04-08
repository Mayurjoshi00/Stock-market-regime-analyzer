

# 📈 Market Regime Detection using HMM

Hidden Markov Model based **market regime detection** with a **risk-on / risk-off trading strategy** using S&P 500 and VIX data.

This project identifies market conditions like:

* 🟢 Strong Bull
* 🟩 Bull
* 🟠 Bear / Unstable
* 🔴 Crisis

…and trades **only during favorable regimes**.

---

# 🧠 How It Works

### 1. Data Collection

* S&P 500 from **STOOQ**
* VIX from **FRED**
* Merged into a single dataset

### 2. Feature Engineering

The model uses:

* Log returns
* Rolling return (20-day)
* Rolling volatility (20-day)
* Drawdown
* VIX level
* VIX percentage change

### 3. Model Training

A **Gaussian Hidden Markov Model** is trained with:

* 4 regimes
* full covariance
* 1000 iterations

### 4. Regime Classification

Model automatically classifies market into:

| Regime | Meaning         |
| ------ | --------------- |
| 0      | Bull            |
| 1      | Strong Bull     |
| 2      | Bear / Unstable |
| 3      | Crisis          |

---

# 📊 Trading Strategy

Risk-On / Risk-Off logic:

* Trade only in **Bull & Strong Bull**
* Stay in cash during **Bear & Crisis**

```
risk_on_regimes = [0, 1]
```

This reduces drawdowns and improves risk-adjusted returns.

---

# 📈 Performance Metrics

The script calculates:

* CAGR
* Volatility
* Sharpe Ratio
* Maximum Drawdown

Comparison:

* Strategy vs Buy & Hold
* Equity curve visualization
* Regime colored chart

---

# 📦 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib hmmlearn pandas_datareader
```

---

# ▶️ Run

Just run:

```bash
python market_regime.py
```

The script will:

1. Download data
2. Train HMM
3. Detect regimes
4. Backtest strategy
5. Print performance
6. Plot charts

---

# 📉 Output

### 1. Strategy vs Buy & Hold

Equity curve comparison

### 2. Market Regimes Plot

Colored dots showing:

* Bull markets
* Bear markets
* Crisis periods

---

# 🧪 Example Use Cases

* Quant trading research
* Market timing strategies
* Risk management overlay
* Portfolio allocation switching
* Volatility regime detection

---

# ⚙️ Model Details

| Parameter  | Value        |
| ---------- | ------------ |
| Model      | Gaussian HMM |
| Regimes    | 4            |
| Features   | 6            |
| Iterations | 1000         |
| Covariance | Full         |

---

# 🧠 Strategy Idea

Instead of holding through crashes:

* Invest in good regimes
* Sit out chaos
* Avoid emotional damage
* Let math do the driving

Basically:
"Trade when market is sane. Hide when it's caffeinated."

---

# 📁 Project Structure

```
market_regime.py
README.md
```

---

# 🚀 Future Improvements

* Add transaction costs
* Walk-forward validation
* Use more features (macro data)
* Multi-asset regime switching
* Live trading integration
* Portfolio allocation per regime

---

# 📜 License

Free to use for research and learning.

