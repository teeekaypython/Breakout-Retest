import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# === CONFIG ===
SYMBOLS = [
    "XAUUSD", "BTCUSD", "Volatility 75 Index", "Volatility 100 Index", "BTCUSD", "ETHUSD",
    "Jump 25 Index", "Jump 50 Index", "Jump 75 Index", "Jump 100 Index",
    "Boom 500 Index", "Crash 300 Index", "Range Break 100 Index",
    "Volatility 15 (1s) Index", "Volatility 50 (1s) Index",
    "Volatility 100 (1s) Index", "Gold RSI Pullback Index",
    "XPBUSD", "DEX 1500 UP Index", "China H Shares", "Japan 225",
    "Vol over Crash 400", "Vol over Crash 750", "Vol over Boom 400"
]
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 5000

INITIAL_BALANCE = 10_000.0
RISK_PER_TRADE = 0.01      # 1% per trade
RR = 2.0                   # 1:2 Reward/Risk
LOOKBACK = 40              # bars to calculate zone
RETTEST_LOOKAHEAD = 20     # bars to wait for retest

# === INITIALIZE MT5 ===
# Uses already logged-in account; no explicit login parameters needed
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed; ensure terminal is running and logged in")

# === FETCH DATA ===
def get_data(symbol):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# === PERFORMANCE CALCULATIONS ===
def compute_statistics(equity, timestamps):
    eq = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    total_ret = (eq[-1] / eq[0]) - 1
    years = (timestamps[-1] - timestamps[0]).total_seconds() / (365.25 * 24 * 3600)
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    avg, std = rets.mean(), rets.std(ddof=1)
    trades_per_year = len(rets) / years if years > 0 else np.nan
    sharpe = (avg / std) * sqrt(trades_per_year) if std > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        'Total Return (%)': round(total_ret * 100, 2),
        'Annual Return (%)': round(ann_ret * 100, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_dd * 100, 2),
        'Calmar Ratio': round(calmar, 2),
        'Trades': len(rets)
    }

# === STRATEGY: BREAKOUT & RETEST ===
def detect_breakout_retests(df):
    df['Signal'] = None
    for i in range(LOOKBACK, len(df) - RETTEST_LOOKAHEAD):
        zone_high = df['high'].iloc[i-LOOKBACK:i].max()
        zone_low = df['low'].iloc[i-LOOKBACK:i].min()
        close_i = df['close'].iloc[i]

        # breakout above
        if close_i > zone_high:
            for j in range(i + 1, i + 1 + RETTEST_LOOKAHEAD):
                if df['low'].iloc[j] <= zone_high:
                    df.at[df.index[j], 'Signal'] = 'buy'
                    break
        # breakout below
        elif close_i < zone_low:
            for j in range(i + 1, i + 1 + RETTEST_LOOKAHEAD):
                if df['high'].iloc[j] >= zone_low:
                    df.at[df.index[j], 'Signal'] = 'sell'
                    break
    return df

# === BACKTEST ENGINE ===
def backtest(df):
    balance = INITIAL_BALANCE
    equity, times = [balance], [df.index[LOOKBACK]]
    wins = losses = 0

    for i in range(LOOKBACK + 1, len(df)):
        sig = df['Signal'].iloc[i]
        price = df['close'].iloc[i]
        if sig in ('buy', 'sell'):
            risk = balance * RISK_PER_TRADE
            sl = df['low'].iloc[i-LOOKBACK:i].min() if sig == 'buy' else df['high'].iloc[i-LOOKBACK:i].max()
            tp = price + (price - sl) * RR if sig == 'buy' else price - (sl - price) * RR

            for j in range(i + 1, len(df)):
                h, l, c = df['high'].iloc[j], df['low'].iloc[j], df['close'].iloc[j]
                if sig == 'buy' and l <= sl:
                    balance -= risk; losses += 1; break
                if sig == 'buy' and c >= tp:
                    balance += risk * RR; wins += 1; break
                if sig == 'sell' and h >= sl:
                    balance -= risk; losses += 1; break
                if sig == 'sell' and c <= tp:
                    balance += risk * RR; wins += 1; break

            equity.append(balance)
            times.append(df.index[j])

    stats = compute_statistics(equity, times)
    return equity, wins, losses, stats

# === EXECUTION ===
for sym in SYMBOLS:
    try:
        df = get_data(sym)
        df = detect_breakout_retests(df)
        eq, w, l, stats = backtest(df)
        print(f"\n=== {sym} ===")
        print(f"Trades: {w + l}, Wins: {w}, Losses: {l}, Win Rate: {round(w/(w+l)*100, 2) if w+l > 0 else 0}%")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        plt.plot(eq, label=sym)
        plt.title(f"Equity Curve — {sym}")
        plt.xlabel("Trades"); plt.ylabel("Balance"); plt.grid(True); plt.legend(); plt.show()
    except Exception as e:
        print(f"Error on {sym}: {e}")

mt5.shutdown()
