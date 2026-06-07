from __future__ import annotations

import hashlib
import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "trading_repository.sqlite3"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

DATE_CANDIDATES = [
    "date", "day", "opendate", "open date", "trade date", "closedate", "close date"
]
PL_CANDIDATES = [
    "daily_pl", "daily p/l", "daily pnl", "p/l", "pl", "pnl", "net p/l", "net pnl",
    "totalnetprofitloss", "total net profit loss", "profit", "profit/loss", "net profit/loss"
]
STRATEGY_CANDIDATES = ["strategy", "system", "setup", "tag", "model"]
TRADE_COUNT_CANDIDATES = ["trades", "trade count", "number of trades", "count"]


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                trader_alias TEXT NOT NULL,
                discord_handle TEXT,
                strategy_name TEXT,
                account_size REAL,
                notes TEXT,
                show_in_group INTEGER NOT NULL DEFAULT 1,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                row_count INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT NOT NULL,
                trader_alias TEXT NOT NULL,
                discord_handle TEXT,
                trade_date TEXT NOT NULL,
                strategy TEXT,
                daily_pl REAL NOT NULL,
                trade_count INTEGER,
                source_row_number INTEGER,
                show_in_group INTEGER NOT NULL DEFAULT 1,
                UNIQUE(upload_id, source_row_number),
                FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
            )
            """
        )
        for table, columns in {
            "uploads": {
                "discord_handle": "TEXT",
                "strategy_name": "TEXT",
                "account_size": "REAL",
                "notes": "TEXT",
                "show_in_group": "INTEGER NOT NULL DEFAULT 1",
            },
            "daily_results": {
                "discord_handle": "TEXT",
                "show_in_group": "INTEGER NOT NULL DEFAULT 1",
            },
        }.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, definition in columns.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")
        conn.commit()


def clean_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ").replace("&", " ")


def find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    cleaned = {clean_col(c): c for c in columns}
    candidate_set = {clean_col(c) for c in candidates}
    for c in candidate_set:
        if c in cleaned:
            return cleaned[c]
    for cleaned_name, original in cleaned.items():
        if any(c in cleaned_name for c in candidate_set):
            return original
    return None


def parse_money(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .astype(float)
    )


def normalize_tradesteward_csv(df: pd.DataFrame, fallback_strategy: str) -> Tuple[pd.DataFrame, dict]:
    columns = list(df.columns)
    date_col = find_column(columns, DATE_CANDIDATES)
    pl_col = find_column(columns, PL_CANDIDATES)
    strategy_col = find_column(columns, STRATEGY_CANDIDATES)
    trade_count_col = find_column(columns, TRADE_COUNT_CANDIDATES)

    if not date_col or not pl_col:
        raise ValueError(
            "Could not identify the required date and P/L columns. Expected something like "
            "OpenDate/Day/Date and TotalNetProfitLoss/Daily_PL/P&L."
        )

    normalized = pd.DataFrame()
    normalized["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
    normalized["daily_pl"] = parse_money(df[pl_col])

    if strategy_col:
        normalized["strategy"] = df[strategy_col].astype(str).replace({"nan": fallback_strategy})
    else:
        normalized["strategy"] = fallback_strategy or "Unspecified"

    if trade_count_col:
        normalized["trade_count"] = pd.to_numeric(df[trade_count_col], errors="coerce").fillna(0).astype(int)
    else:
        normalized["trade_count"] = np.nan

    normalized["source_row_number"] = np.arange(1, len(df) + 1)
    normalized = normalized.dropna(subset=["trade_date", "daily_pl"])
    normalized = normalized[normalized["trade_date"] != "NaT"]

    mapping = {
        "date_col": date_col,
        "pl_col": pl_col,
        "strategy_col": strategy_col or "Not found; using strategy entered on upload form",
        "trade_count_col": trade_count_col or "Not found",
    }
    return normalized, mapping


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_alias(discord_handle: str, anonymous: bool) -> str:
    handle = (discord_handle or "").strip()
    if anonymous or not handle:
        digest = hashlib.sha256((handle + datetime.utcnow().isoformat()).encode()).hexdigest()[:6].upper()
        return f"Trader_{digest}"
    return handle if handle.startswith("@") else f"@{handle}"


def save_upload(
    trader_alias: str,
    discord_handle: str,
    strategy_name: str,
    account_size: Optional[float],
    notes: str,
    show_in_group: bool,
    uploaded_file,
) -> Tuple[str, pd.DataFrame, dict]:
    content = uploaded_file.getvalue()
    digest = file_sha256(content)
    upload_id = digest[:16]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stored_filename = f"{timestamp}_{upload_id}_{uploaded_file.name}"
    stored_path = UPLOAD_DIR / stored_filename
    stored_path.write_bytes(content)

    raw_df = pd.read_csv(io.BytesIO(content))
    normalized, mapping = normalize_tradesteward_csv(raw_df, strategy_name or "Unspecified")
    show_flag = 1 if show_in_group else 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO uploads
            (upload_id, trader_alias, discord_handle, strategy_name, account_size, notes, show_in_group,
             original_filename, stored_filename, uploaded_at, file_hash, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                trader_alias,
                discord_handle,
                strategy_name,
                account_size,
                notes,
                show_flag,
                uploaded_file.name,
                stored_filename,
                datetime.utcnow().isoformat(),
                digest,
                len(raw_df),
            ),
        )
        records = normalized.assign(
            upload_id=upload_id,
            trader_alias=trader_alias,
            discord_handle=discord_handle,
            show_in_group=show_flag,
        )[
            [
                "upload_id",
                "trader_alias",
                "discord_handle",
                "trade_date",
                "strategy",
                "daily_pl",
                "trade_count",
                "source_row_number",
                "show_in_group",
            ]
        ].to_records(index=False)
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_results
            (upload_id, trader_alias, discord_handle, trade_date, strategy, daily_pl, trade_count, source_row_number, show_in_group)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(records),
        )
        conn.commit()

    return upload_id, normalized, mapping


def load_daily_results(public_only: bool = True) -> pd.DataFrame:
    where = "WHERE show_in_group = 1" if public_only else ""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM daily_results {where}", conn)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["daily_pl"] = pd.to_numeric(df["daily_pl"], errors="coerce")
    return df.dropna(subset=["trade_date", "daily_pl"])


def load_uploads() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM uploads ORDER BY uploaded_at DESC", conn)
    return df


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()
    out["cumulative_pl"] = out["daily_pl"].cumsum()
    out["running_peak"] = out["cumulative_pl"].cummax()
    out["drawdown"] = out["cumulative_pl"] - out["running_peak"]
    return out


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for trader, g in df.groupby("trader_alias"):
        s = g["daily_pl"].dropna()
        losses = abs(s[s < 0].sum())
        profit_factor = s[s > 0].sum() / losses if losses > 0 else np.nan
        m = add_metrics(g)
        rows.append(
            {
                "trader_alias": trader,
                "days": int(s.count()),
                "total_pl": float(s.sum()),
                "avg_day": float(s.mean()),
                "median_day": float(s.median()),
                "std_day": float(s.std(ddof=1)) if s.count() > 1 else np.nan,
                "best_day": float(s.max()),
                "worst_day": float(s.min()),
                "win_rate": float((s > 0).mean()) if s.count() else np.nan,
                "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
                "max_drawdown": float(m["drawdown"].min()) if not m.empty else np.nan,
                "first_day": g["trade_date"].min().date(),
                "last_day": g["trade_date"].max().date(),
            }
        )
    return pd.DataFrame(rows).sort_values("total_pl", ascending=False)


def community_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("trade_date", as_index=False)["daily_pl"].sum().sort_values("trade_date")
    out["cumulative_pl"] = out["daily_pl"].cumsum()
    out["rolling_20_day_return"] = out["daily_pl"].rolling(20, min_periods=5).sum()
    out["rolling_20_day_vol"] = out["daily_pl"].rolling(20, min_periods=5).std()
    out["ewma_vol"] = out["daily_pl"].ewm(span=20, adjust=False).std()
    return out


def build_projection(daily: pd.DataFrame, horizon_days: int = 21, sims: int = 5000) -> pd.DataFrame:
    """One-month projection cone from historical mean and recent EWMA volatility of daily P/L."""
    s = daily["daily_pl"].dropna().astype(float)
    if len(s) < 5:
        return pd.DataFrame()

    mu = float(s.tail(60).mean()) if len(s) >= 20 else float(s.mean())
    hist_vol = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    ewma_vol = float(s.ewm(span=20, adjust=False).std().iloc[-1]) if len(s) > 3 else hist_vol
    sigma = ewma_vol if np.isfinite(ewma_vol) and ewma_vol > 0 else hist_vol
    if not np.isfinite(sigma) or sigma <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(42)
    simulated_daily = rng.normal(loc=mu, scale=sigma, size=(sims, horizon_days))
    paths = simulated_daily.cumsum(axis=1)
    quantiles = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)

    last_date = pd.to_datetime(daily["trade_date"].max())
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
    projection = pd.DataFrame(
        {
            "date": future_dates,
            "p10": quantiles[0],
            "p25": quantiles[1],
            "median": quantiles[2],
            "p75": quantiles[3],
            "p90": quantiles[4],
        }
    )
    return projection


def projection_chart(proj: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=proj["date"], y=proj["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj["date"], y=proj["p10"], mode="lines", fill="tonexty", line=dict(width=0),
            name="10-90% range", hovertemplate="%{x|%b %d}<br>10th pct: $%{y:,.0f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj["date"], y=proj["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj["date"], y=proj["p25"], mode="lines", fill="tonexty", line=dict(width=0),
            name="25-75% range", hovertemplate="%{x|%b %d}<br>25th pct: $%{y:,.0f}<extra></extra>"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=proj["date"], y=proj["median"], mode="lines+markers", name="Median projection",
            hovertemplate="%{x|%b %d}<br>Median: $%{y:,.0f}<extra></extra>"
        )
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Projected 1-Month Community P/L",
        xaxis_title="Projected trading day",
        yaxis_title="Projected cumulative P/L",
        hovermode="x unified",
        legend_title_text="Projection band",
    )
    return fig


st.set_page_config(page_title="ALGO Edge Performance History", layout="wide")
init_db()

st.title("ALGO Edge Performance History")
st.caption("Open upload portal for TradeSteward-style CSVs. No login required. Use a Discord handle or stay anonymous.")

with st.sidebar:
    st.header("Upload Trading History")
    discord_handle = st.text_input("Discord handle", placeholder="@your_handle")
    anonymous = st.checkbox("Upload anonymously", value=False)
    strategy_name = st.text_input("Strategy name", placeholder="0DTE / systematic options / etc.")
    account_size = st.number_input("Approx. account size (optional)", min_value=0.0, value=0.0, step=1000.0)
    notes = st.text_area("Notes / setup description", placeholder="Optional: time period, sizing, symbols, risk rules...")
    show_in_group = st.checkbox("Show my results in the group dashboard", value=True)
    uploaded = st.file_uploader("Upload TradeSteward CSV", type=["csv"])

    if uploaded and st.button("Process upload", type="primary"):
        try:
            alias = safe_alias(discord_handle, anonymous)
            acct = account_size if account_size > 0 else None
            upload_id, normalized, mapping = save_upload(
                alias,
                discord_handle.strip(),
                strategy_name.strip() or "Unspecified",
                acct,
                notes.strip(),
                show_in_group,
                uploaded,
            )
            st.success(f"Upload processed: {upload_id}")
            st.write("Detected columns:", mapping)
            st.dataframe(normalized.head(25), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.warning("Before uploading, remove account numbers, brokerage IDs, addresses, or other personal identifiers.")

public_results = load_daily_results(public_only=True)
uploads_df = load_uploads()

if public_results.empty:
    st.info("Upload a CSV to begin. The app expects a date column and a daily P/L column.")
    if not uploads_df.empty:
        st.subheader("Private/hidden uploads exist")
        st.write("Some uploads may be hidden from the group dashboard because the uploader unchecked the visibility box.")
    st.stop()

community = community_daily(public_results)
projection = build_projection(community)

st.subheader("Projected 1-Month Return Outlook")
if projection.empty:
    st.info("Need at least 5 visible trading days to create a projection cone.")
else:
    st.plotly_chart(projection_chart(projection), use_container_width=True)
    end = projection.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Median 1-month projection", f"${end['median']:,.0f}")
    c2.metric("Interquartile range", f"${end['p25']:,.0f} to ${end['p75']:,.0f}")
    c3.metric("Wide range", f"${end['p10']:,.0f} to ${end['p90']:,.0f}")
    st.caption("Projection uses historical daily community P/L, recent EWMA volatility, and 5,000 simulated 21-trading-day paths. It is not a prediction or recommendation.")

st.subheader("Community Performance Dashboard")
metric_cols = st.columns(4)
metric_cols[0].metric("Visible traders", public_results["trader_alias"].nunique())
metric_cols[1].metric("Total trading days", f"{len(public_results):,}")
metric_cols[2].metric("Group net P/L", f"${public_results['daily_pl'].sum():,.0f}")
metric_cols[3].metric("Uploads", f"{len(uploads_df):,}")

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

if filtered.empty:
    st.warning("Select at least one trader to view the dashboard.")
    st.stop()

left, right = st.columns([2, 1])
with right:
    st.markdown("### Summary")
    stats = summary_stats(filtered)
    st.dataframe(
        stats.style.format({
            "total_pl": "${:,.0f}",
            "avg_day": "${:,.0f}",
            "median_day": "${:,.0f}",
            "std_day": "${:,.0f}",
            "best_day": "${:,.0f}",
            "worst_day": "${:,.0f}",
            "win_rate": "{:.1%}",
            "profit_factor": "{:.2f}",
            "max_drawdown": "${:,.0f}",
        }),
        use_container_width=True,
    )

with left:
    st.markdown("### Cumulative Returns")
    curve_df = pd.concat([add_metrics(g) for _, g in filtered.groupby("trader_alias")], ignore_index=True)
    fig = px.line(curve_df, x="trade_date", y="cumulative_pl", color="trader_alias", title="Cumulative P/L")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Volatility Trend")
filtered_community = community_daily(filtered)
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=filtered_community["trade_date"], y=filtered_community["rolling_20_day_vol"], mode="lines", name="20-day realized volatility"))
fig_vol.add_trace(go.Scatter(x=filtered_community["trade_date"], y=filtered_community["ewma_vol"], mode="lines", name="EWMA volatility forecast"))
fig_vol.update_layout(title="Daily P/L Volatility", xaxis_title="Date", yaxis_title="Daily P/L volatility", hovermode="x unified")
st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("### Rolling 20-Day Return")
fig_roll = px.line(filtered_community, x="trade_date", y="rolling_20_day_return", title="Rolling 20-Day Community P/L")
st.plotly_chart(fig_roll, use_container_width=True)

st.markdown("### Daily P/L Distribution")
fig_hist = px.histogram(filtered, x="daily_pl", color="trader_alias", nbins=60, marginal="box", title="Daily P/L Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("Recent uploads"):
    display_cols = ["uploaded_at", "trader_alias", "strategy_name", "row_count", "show_in_group", "original_filename"]
    display_uploads = uploads_df[[c for c in display_cols if c in uploads_df.columns]].copy()
    st.dataframe(display_uploads, use_container_width=True)

with st.expander("Export visible standardized dataset"):
    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: this version stores data locally in the Streamlit app container. For durable Discord group use, move the database/storage to Supabase next.")
