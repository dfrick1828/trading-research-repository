# Trading Research Repository

A Streamlit MVP for a private trading group to upload TradeSteward-style trading history CSVs, normalize daily P/L data, and analyze individual and group performance.

## What it does

- Upload CSV files from TradeSteward or similar exports
- Map common column names like `OpenDate`, `Day`, `Date`, `TotalNetProfitLoss`, `Daily_PL`, `P/L`, `Net P/L`
- Store raw uploads locally in `/data/uploads`
- Store normalized daily results in a local SQLite database
- Generate:
  - Equity curves
  - Drawdown curves
  - Return distribution histogram
  - Group summary statistics
  - Trader comparison table

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Important MVP notes

This first version uses local SQLite and local file storage. That is good for prototyping but not ideal for a live multi-user group. The next production step is Supabase/Postgres with authenticated users and row-level security.

## Privacy

Use aliases like `Trader_001`, not real names. Avoid uploading account numbers, brokerage IDs, or personally identifying data.
