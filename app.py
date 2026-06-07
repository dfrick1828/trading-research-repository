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
TRADE_COUNT_CANDIDATES = ["trades", "trade count", "number of trades", "count"]


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                trader_alias TEXT NOT NULL,
                discord_handle TEXT,
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
                daily_pl REAL NOT NULL,
                trade_count INTEGER,
                source_row_number INTEGER,
                show_in_group INTEGER NOT NULL DEFAULT 1,
                UNIQUE(upload_id, source_row_number),
                FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
            )
            """
        )
        # migrations from prior builds
        migrations = {
            "uploads": {
                "discord_handle": "TEXT",
                "account_size": "REAL",
                "notes": "TEXT",
                "show_in_group": "INTEGER NOT NULL DEFAULT 1",
            },
            "daily_results": {
                "discord_handle": "TEXT",
                "show_in_group": "INTEGER NOT NULL DEFAULT 1",
            },
        }
        for table, columns in migrations.items():
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
    for candidate in candidate_set:
        if candidate in cleaned:
            return cleaned[candidate]
    for cleaned_name, original in cleaned.items():
        if any(candidate in cleaned_name for candidate in candidate_set):
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


def normalize_csv(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    columns = list(df.columns)
    date_col = find_column(columns, DATE_CANDIDATES)
    pl_col = find_column(columns, PL_CANDIDATES)
    trade_count_col = find_column(columns, TRADE_COUNT_CANDIDATES)

    if not date_col or not pl_col:
        raise ValueError(
            "Could not identify required columns. Expected a date column like OpenDate/Day/Date "
            "and a P/L column like TotalNetProfitLoss/Daily_PL/P&L."
        )

    out = pd.DataFrame()
    out["trade_date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["daily_pl"] = parse_money(df[pl_col])
    if trade_count_col:
        out["trade_count"] = pd.to_numeric(df[trade_count_col], errors="coerce")
    else:
        out["trade_count"] = np.nan
    out["source_row_number"] = np.arange(1, len(df) + 1)
    out = out.dropna(subset=["trade_date", "daily_pl"]).copy()
    out["trade_date"] = out["trade_date"].dt.date.astype(str)

    mapping = {
        "date_col": date_col,
        "pl_col": pl_col,
        "trade_count_col": trade_count_col or "Not found",
    }
    return out, mapping


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
    (UPLOAD_DIR / stored_filename).write_bytes(content)

    raw_df = pd.read_csv(io.BytesIO(content))
    normalized, mapping = normalize_csv(raw_df)
    show_flag = 1 if show_in_group else 0

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO uploads
            (upload_id, trader_alias, discord_handle, account_size, notes, show_in_group,
             original_filename, stored_filename, uploaded_at, file_hash, row_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                trader_alias,
                discord_handle,
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
                "daily_pl",
                "trade_count",
                "source_row_number",
                "show_in_group",
            ]
        ].to_records(index=False)
        conn.executemany(
            """
            INSERT OR REPLACE INTO daily_results
            (upload_id, trader_alias, discord_handle, trade_date, daily_pl, trade_count, source_row_number, show_in_group)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["daily_pl"] = pd.to_numeric(df["daily_pl"], errors="coerce")
    return df.dropna(subset=["trade_date", "daily_pl"])


def load_uploads() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM uploads ORDER BY uploaded_at DESC", conn)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()
    out["cumulative_pl"] = out["daily_pl"].cumsum()
    out["running_peak"] = out["cumulative_pl"].cummax()
    out["drawdown"] = out["cumulative_pl"] - out["running_peak"]
    out["rolling_20_day_pl"] = out["daily_pl"].rolling(20, min_periods=5).sum()
    out["rolling_20_day_vol"] = out["daily_pl"].rolling(20, min_periods=5).std()
    return out


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for trader, g in df.groupby("trader_alias"):
        s = g["daily_pl"].dropna()
        gains = s[s > 0].sum()
        losses = abs(s[s < 0].sum())
        metrics = add_metrics(g)
        rows.append({
            "trader_alias": trader,
            "days": int(len(s)),
            "total_pl": float(s.sum()),
            "avg_day": float(s.mean()),
            "median_day": float(s.median()),
            "std_day": float(s.std()) if len(s) > 1 else 0.0,
            "best_day": float(s.max()),
            "worst_day": float(s.min()),
            "win_rate": float((s > 0).mean()),
            "profit_factor": float(gains / losses) if losses > 0 else np.nan,
            "max_drawdown": float(metrics["drawdown"].min()),
            "first_day": g["trade_date"].min().date(),
            "last_day": g["trade_date"].max().date(),
        })
    return pd.DataFrame(rows).sort_values("total_pl", ascending=False)


st.set_page_config(page_title="ALGO Edge Performance History", layout="wide")
init_db()

st.title("ALGO Edge Performance History")
st.caption("Simple Discord upload portal and return dashboard for systematic traders.")

with st.sidebar:
    st.header("Upload Trading History")
    discord_handle = st.text_input("Discord handle", placeholder="@your_handle")
    anonymous = st.checkbox("Upload anonymously", value=False)
    account_size = st.number_input("Approx. account size (optional)", min_value=0.0, value=0.0, step=1000.0)
    notes = st.text_area("Notes / setup description", placeholder="Optional: time period, sizing, risk rules...")
    show_in_group = st.checkbox("Show my results in the group dashboard", value=True)
    uploaded = st.file_uploader("Upload TradeSteward CSV", type=["csv"])

    if uploaded and st.button("Process upload", type="primary"):
        try:
            alias = safe_alias(discord_handle, anonymous)
            acct = account_size if account_size > 0 else None
            upload_id, normalized, mapping = save_upload(
                alias,
                discord_handle.strip(),
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
    st.warning("Remove account numbers, brokerage IDs, addresses, and personal identifiers before uploading.")

public_results = load_daily_results(public_only=True)
uploads_df = load_uploads()

if public_results.empty:
    st.info("Upload a CSV to begin. The app expects a date column and a daily P/L column.")
    st.stop()

st.subheader("Community Return Dashboard")

metric_cols = st.columns(4)
metric_cols[0].metric("Visible traders", public_results["trader_alias"].nunique())
metric_cols[1].metric("Trading days", f"{len(public_results):,}")
metric_cols[2].metric("Group net P/L", f"${public_results['daily_pl'].sum():,.0f}")
metric_cols[3].metric("Uploads", f"{len(uploads_df):,}")

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

if filtered.empty:
    st.warning("Select at least one trader.")
    st.stop()

curve_df = pd.concat([add_metrics(g) for _, g in filtered.groupby("trader_alias")], ignore_index=True)

left, right = st.columns([2, 1])
with left:
    st.markdown("### Cumulative Returns")
    fig = px.line(curve_df, x="trade_date", y="cumulative_pl", color="trader_alias", title="Cumulative P/L")
    st.plotly_chart(fig, use_container_width=True)

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

st.markdown("### Rolling 20-Day Returns")
roll_df = curve_df.dropna(subset=["rolling_20_day_pl"])
if roll_df.empty:
    st.info("Need at least 5 trading days per trader to show rolling returns.")
else:
    fig_roll = px.line(roll_df, x="trade_date", y="rolling_20_day_pl", color="trader_alias", title="Rolling 20-Day P/L")
    st.plotly_chart(fig_roll, use_container_width=True)

st.markdown("### Daily P/L Distribution")
fig_hist = px.histogram(filtered, x="daily_pl", color="trader_alias", nbins=60, marginal="box", title="Daily P/L Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("### Volatility Trend")
vol_df = curve_df.dropna(subset=["rolling_20_day_vol"])
if vol_df.empty:
    st.info("Need at least 5 trading days per trader to show volatility trend.")
else:
    fig_vol = px.line(vol_df, x="trade_date", y="rolling_20_day_vol", color="trader_alias", title="Rolling 20-Day Daily P/L Volatility")
    st.plotly_chart(fig_vol, use_container_width=True)

with st.expander("Recent uploads"):
    cols = ["uploaded_at", "trader_alias", "row_count", "show_in_group", "original_filename"]
    st.dataframe(uploads_df[[c for c in cols if c in uploads_df.columns]], use_container_width=True)

with st.expander("Export visible standardized dataset"):
    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: this version stores data locally in the Streamlit app container. For durable group use, move storage to Supabase later.")
