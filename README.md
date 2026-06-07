# Discord Trading Research Repository

A no-login Streamlit app for Discord trading groups to upload TradeSteward-style CSV exports and view shared analytics.

## Features

- No login required
- Discord handle field
- Anonymous upload option
- Strategy name field
- Account size and notes fields
- Toggle to show/hide results from the group dashboard
- Standardizes TradeSteward-style date and P/L columns
- Group equity curves
- Drawdown curves
- Daily P/L histograms
- Trader summary statistics
- Strategy breakdown
- Downloadable standardized group dataset

## GARCH Volatility Dashboard

The app now includes a GARCH-style volatility dashboard that:

- Aggregates selected traders into one group book
- Converts daily P/L to daily returns using uploaded account size, user override, or inferred capital base
- Fits a simple zero-mean GARCH(1,1) model using maximum likelihood when enough observations exist
- Falls back to stable default parameters when the sample is too small
- Displays current volatility regime, annualized forecast volatility, next-day forecast volatility, and persistence
- Compares GARCH volatility with 5-day and 20-day realized volatility
- Shows return bars overlaid with forecast volatility
- Summarizes performance in low, normal, and high volatility regimes

## Deploy on Streamlit

Main file path:

```text
app.py
```

Requirements are in `requirements.txt`.

## Important MVP limitation

This version stores data in local SQLite and local upload files. On Streamlit Community Cloud, local storage can reset when the app restarts or redeploys. For real group use, move storage to Supabase.
