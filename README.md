# ALGO Edge Performance History

A no-login Streamlit app for Discord trading groups to upload TradeSteward-style CSV exports and view shared performance, volatility, and one-month projected return analytics.

## Features

- No login required
- Discord handle field
- Anonymous upload option
- Optional account size for percentage-return normalization
- Toggle to show/hide results from the group dashboard
- Standardizes TradeSteward-style date and P/L columns
- Front-page projected 1-month return graph
- Monte Carlo projection using historical return volatility
- EWMA/GARCH-style volatility forecast
- Volatility trend dashboard
- Cumulative returns
- Rolling 20-day returns
- Daily return/P&L distribution
- Trader summary statistics
- Downloadable standardized group dataset

## Deploy on Streamlit

Main file path:

```text
app.py
```

Requirements are in `requirements.txt`.

## Important MVP limitation

This version stores data in local SQLite and local upload files. On Streamlit Community Cloud, local storage can reset when the app restarts or redeploys. For real group use, move storage to Supabase.
