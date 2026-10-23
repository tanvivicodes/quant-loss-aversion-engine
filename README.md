# Quant Loss Aversion Engine

This is a Python tool I built to track stock market panic. Most trading models assume investors make perfectly rational decisions, but real people panic-sell when the market crashes. 

This engine uses `yfinance` to pull real-time data and tracks sharp volume spikes during price drops. I added a biological decay function to model how human fear stays high after a market shock, slowly cooling down over a few days. 

When the fear metrics hit a statistical extreme, the script triggers a buy-the-dip signal.

## How to run it:
1. Run `pip install yfinance pandas numpy scipy matplotlib`
2. Run `python quant-loss-aversion-engine.py`
