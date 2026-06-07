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

## Deploy on Streamlit

Main file path:

```text
app.py
```

Requirements are in `requirements.txt`.

## Important MVP limitation

This version stores data in local SQLite and local upload files. On Streamlit Community Cloud, local storage can reset when the app restarts or redeploys. For real group use, the next upgrade should move storage to Supabase.
