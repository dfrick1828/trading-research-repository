# ALGO Edge Performance History

A no-login Streamlit app for Discord trading groups to upload TradeSteward-style CSV exports and view shared performance analytics.

## Features

- No login required
- Discord handle field
- Anonymous upload option
- Toggle to show/hide results from the group dashboard
- Standardizes TradeSteward-style date and P/L columns
- Front-page projected 1-month return cone
- GARCH-style / EWMA volatility trend
- Cumulative return history
- Rolling return trend
- Daily return distribution
- Trader summary statistics
- Downloadable standardized group dataset

## Deploy on Streamlit

Main file path:

```text
app.py
```

Requirements are in `requirements.txt`.

## Projection note

The 1-month projection uses historical daily P/L and a dependency-light EWMA/GARCH-style volatility estimate. It is a scenario cone, not a prediction or trading recommendation.

## Important MVP limitation

This version stores data in local SQLite and local upload files. On Streamlit Community Cloud, local storage can reset when the app restarts or redeploys. For real group use, move storage to Supabase.
