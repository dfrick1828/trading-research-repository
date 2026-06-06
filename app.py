from __future__ import annotations

import hashlib
import io
import os
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
    "totalnetprofitloss", "total net profit loss", "profit", "profit/loss"
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
                trade_date TEXT NOT NULL,
                strategy TEXT,
                daily_pl REAL NOT NULL,
                trade_count INTEGER,
                source_row_number INTEGER,
                UNIQUE(upload_id, source_row_number),
                FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
            )
            """
        )
        conn.commit()


def clean_col(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


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


def normalize_tradesteward_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    columns = list(df.columns)
    date_col = find_column(columns, DATE_CANDIDATES)
    pl_col = find_column(columns, PL_CANDIDATES)
    strategy_col = find_column(columns, STRATEGY_CANDIDATES)
    trade_count_col = find_column(columns, TRADE_COUNT_CANDIDATES)

    if not date_col or not pl_col:
        raise ValueError(
            "Could not identify the required date and P/L columns. "
            "Expected something like OpenDate/Day/Date and TotalNetProfitLoss/Daily_PL/P&L."
        )

    normalized = pd.DataFrame()
    normalized["trade_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
    normalized["daily_pl"] = parse_money(df[pl_col])
    normalized["strategy"] = df[strategy_col].astype(str) if strategy_col else "Unspecified"

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
        "strategy_col": strategy_col or "Not found; set to Unspecified",
        "trade_count_col": trade_count_col or "Not found",
    }
    return normalized, mapping


def file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_upload(trader_alias: str, uploaded_file) -> Tuple[str, pd.DataFrame, dict]:
    content = uploaded_file.getvalue()
    digest = file_sha256(content)
    upload_id = digest[:16]
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stored_filename = f"{timestamp}_{upload_id}_{uploaded_file.name}"
    stored_path = UPLOAD_DIR / stored_filename
    stored_path.write_bytes(content)

    raw_df = pd.read_csv(io.BytesIO(content))
    normalized, mapping = normalize_tradesteward_csv(raw_df)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO uploads
            (upload_id, trader_alias, original_filename, stored_filename, uploaded_at, file_hash, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (upload_id, trader_alias, uploaded_file.name, stored_filename, datetime.utcnow().isoformat(), digest, len(raw_df)),
        )
        records = normalized.assign(upload_id=upload_id, trader_alias=trader_alias)[
            ["upload_id", "trader_alias", "trade_date", "strategy", "daily_pl", "trade_count", "source_row_number"]
        ].to_records(index=False)
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_results
            (upload_id, trader_alias, trade_date, strategy, daily_pl, trade_count, source_row_number)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            list(records),
        )
        conn.commit()

    return upload_id, normalized, mapping


def load_daily_results() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM daily_results", conn)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["daily_pl"] = pd.to_numeric(df["daily_pl"], errors="coerce")
    return df


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()
    out["cumulative_pl"] = out["daily_pl"].cumsum()
    out["running_peak"] = out["cumulative_pl"].cummax()
    out["drawdown"] = out["cumulative_pl"] - out["running_peak"]
    return out


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("trader_alias")
    stats = grouped["daily_pl"].agg(
        days="count",
        total_pl="sum",
        avg_day="mean",
        median_day="median",
        std_day="std",
        best_day="max",
        worst_day="min",
    ).reset_index()
    stats["win_rate"] = grouped["daily_pl"].apply(lambda s: (s > 0).mean()).values
    stats["profit_factor"] = grouped["daily_pl"].apply(
        lambda s: s[s > 0].sum() / abs(s[s < 0].sum()) if abs(s[s < 0].sum()) > 0 else np.nan
    ).values
    stats["max_drawdown"] = grouped.apply(lambda g: add_metrics(g)["drawdown"].min()).values
    return stats.sort_values("total_pl", ascending=False)


st.set_page_config(page_title="Trading Research Repository", layout="wide")
init_db()

st.title("Trading Research Repository")
st.caption("Private MVP for uploading TradeSteward-style trading history and analyzing individual/group return profiles.")

with st.sidebar:
    st.header("Upload")
    trader_alias = st.text_input("Trader alias", value="Trader_001")
    uploaded = st.file_uploader("Upload TradeSteward CSV", type=["csv"])
    if uploaded and st.button("Process upload", type="primary"):
        try:
            upload_id, normalized, mapping = save_upload(trader_alias.strip() or "Anonymous", uploaded)
            st.success(f"Upload processed: {upload_id}")
            st.write("Detected columns:", mapping)
            st.dataframe(normalized.head(25), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.warning("Do not upload account numbers, brokerage IDs, or personal identifying information.")

all_results = load_daily_results()

if all_results.empty:
    st.info("Upload a CSV to begin. The app expects a date column and a daily P/L column.")
    st.stop()

traders = sorted(all_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Traders", traders, default=traders)
filtered = all_results[all_results["trader_alias"].isin(selected_traders)].copy()

left, right = st.columns([2, 1])
with right:
    st.subheader("Group Summary")
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
    st.subheader("Equity Curves")
    curve_df = []
    for trader, g in filtered.groupby("trader_alias"):
        m = add_metrics(g)
        curve_df.append(m)
    curve_df = pd.concat(curve_df, ignore_index=True)
    fig = px.line(curve_df, x="trade_date", y="cumulative_pl", color="trader_alias", title="Cumulative P/L")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Drawdowns")
fig_dd = px.line(curve_df, x="trade_date", y="drawdown", color="trader_alias", title="Drawdown from Running Peak")
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("Daily P/L Distribution")
fig_hist = px.histogram(filtered, x="daily_pl", color="trader_alias", nbins=60, marginal="box", title="Daily P/L Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Strategy Breakdown")
strategy = filtered.groupby(["trader_alias", "strategy"], dropna=False)["daily_pl"].agg(["count", "sum", "mean", "min", "max"]).reset_index()
st.dataframe(
    strategy.style.format({"sum": "${:,.0f}", "mean": "${:,.0f}", "min": "${:,.0f}", "max": "${:,.0f}"}),
    use_container_width=True,
)
