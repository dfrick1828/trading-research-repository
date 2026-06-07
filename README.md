# ALGO Edge Performance History

Discord-friendly Streamlit app for TradeSteward-style CSV uploads, return-focused analytics, volatility trends, and a front-page 1-month projected return cone.

## Deploy

Main file path:

```text
app.py
```

## Projection note

The 1-month projection uses visible uploaded daily returns, an EWMA/GARCH-style volatility estimate, and Monte Carlo simulation over 21 trading days. It is research output, not investment advice.
