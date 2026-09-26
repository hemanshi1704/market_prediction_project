# Stock Market Prediction: Regime-Aware, Multi-Asset, Confidence-Gated

Predicts next-day directional movement (up/down) for a stock using a model designed
around three ideas, rather than a single "technical indicators + classifier" approach:

- **Regime-aware ensemble** — a Gaussian Mixture Model detects the market's latent
  regime (e.g. calm/trending vs. volatile/choppy) from volatility and trend features,
  and a separate classifier is trained per regime, since the relationship between
  indicators and next-day returns isn't stable across market conditions.
- **Multi-asset signal** — features go beyond the stock's own price history to include
  its sector ETF, the VIX (volatility gauge), a bond ETF (risk-on/risk-off proxy), and
  the S&P 500, plus relative-strength and rolling-correlation features derived from them.
- **Confidence-gated predictions** — the model only "acts" on predictions above a
  probability threshold, and accuracy is reported separately for high-confidence vs.
  all predictions, mirroring how a real trader would only take high-conviction trades.

## Data

Pulled live via [`yfinance`](https://github.com/ranaroussi/yfinance): daily close
prices for the target ticker, its sector ETF, `^VIX`, `TLT`, and `SPY`, from 2015
onward (configurable in the notebook).

## Method

1. Engineer technical features (returns, moving averages, RSI, MACD, Bollinger position,
   rolling volatility) on the target stock, plus cross-asset context features.
2. Detect market regimes with a Gaussian Mixture Model, **refit inside every
   walk-forward fold using only that fold's training data**, to avoid lookahead bias.
3. Train a Gradient Boosting classifier per regime on the training fold; fall back to a
   global model for any regime seen in test but not in training.
4. Evaluate out-of-sample with **walk-forward validation** (not a random train/test
   split — financial data is a time series, and shuffling rows leaks the future).
5. Compare against a naive majority-class baseline and a buy-and-hold backtest.

## Results

<!-- Fill this in after you run the notebook locally. Report the real numbers, not
     placeholders — this section is what people actually read. -->

- Ticker: `TODO`
- Date range: `TODO`
- Out-of-sample accuracy (all predictions): `TODO`
- Naive majority-class baseline: `TODO`
- Out-of-sample accuracy (high-confidence predictions only): `TODO` (coverage: `TODO`)
- Strategy Sharpe ratio vs. buy-and-hold: `TODO`

## Limitations

Next-day price movement is close to a random walk in a reasonably efficient market, so
a small edge over baseline (a few accuracy points) is consistent with noise and isn't
by itself evidence of a tradeable signal. The backtest here is simplified — it ignores
slippage beyond a flat cost assumption, position sizing, liquidity, and taxes. Regimes
are unsupervised clusters, not guaranteed to match labels like "bull" or "bear." See the
notebook's final section for the full discussion, including suggested next steps
(testing across multiple tickers, statistical significance testing, paper trading).

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/market_prediction.ipynb
```

Requires internet access (the notebook pulls live data via `yfinance`). Change
`TICKER` / `SECTOR_ETF` at the top of the notebook to point at a different stock.
