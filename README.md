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
- Volatility Trend dashboard
- Rolling realized volatility
- GARCH-style EWMA forecast volatility
- Volatility regime classification

## Deploy on Streamlit

Main file path:

```text
app.py
```

Requirements are in `requirements.txt`.

## Volatility note

The volatility dashboard uses standardized daily P/L so traders with different account sizes can be compared by volatility regime. The GARCH-style forecast is an EWMA variance recursion designed to be lightweight and reliable on Streamlit Cloud. A future durable version can add account-size-normalized returns and a full GARCH package.

## Important MVP limitation

This version stores data in local SQLite and local upload files. On Streamlit Community Cloud, local storage can reset when the app restarts or redeploys. For real group use, move storage to Supabase.
