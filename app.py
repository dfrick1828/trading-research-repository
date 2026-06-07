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
        # Lightweight migrations for anyone who already deployed the first MVP.
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
    return df


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
    stats["max_drawdown"] = grouped.apply(lambda g: add_metrics(g)["drawdown"].min(), include_groups=False).values
    stats["first_day"] = grouped["trade_date"].min().dt.date.values
    stats["last_day"] = grouped["trade_date"].max().dt.date.values
    return stats.sort_values("total_pl", ascending=False)




def load_public_results_with_uploads() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT d.*, u.account_size, u.uploaded_at
            FROM daily_results d
            LEFT JOIN uploads u ON d.upload_id = u.upload_id
            WHERE d.show_in_group = 1
            """,
            conn,
        )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["daily_pl"] = pd.to_numeric(df["daily_pl"], errors="coerce")
    df["account_size"] = pd.to_numeric(df.get("account_size"), errors="coerce")
    return df.dropna(subset=["trade_date", "daily_pl"])


def add_return_metrics(df: pd.DataFrame, assumed_account_size: float = 25000.0) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()
    acct = pd.to_numeric(out.get("account_size"), errors="coerce")
    out["return_base"] = acct.where(acct > 0, assumed_account_size)
    out["daily_return"] = out["daily_pl"] / out["return_base"]
    out["daily_return"] = out["daily_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Clip extreme data-entry mistakes so one bad upload does not destroy the dashboard.
    out["daily_return"] = out["daily_return"].clip(lower=-0.50, upper=0.50)
    out["growth_index"] = (1.0 + out["daily_return"]).cumprod() * 100.0
    out["cumulative_return_pct"] = out["growth_index"] - 100.0
    out["rolling_20d_return_pct"] = ((1.0 + out["daily_return"]).rolling(20).apply(np.prod, raw=True) - 1.0) * 100.0
    out["rolling_20d_vol_pct"] = out["daily_return"].rolling(20).std() * np.sqrt(252) * 100.0
    return out


def build_community_daily_returns(df: pd.DataFrame, assumed_account_size: float = 25000.0) -> pd.DataFrame:
    pieces = []
    for _, g in df.groupby("trader_alias"):
        pieces.append(add_return_metrics(g, assumed_account_size)[["trade_date", "trader_alias", "daily_return"]])
    if not pieces:
        return pd.DataFrame(columns=["trade_date", "daily_return"])
    all_returns = pd.concat(pieces, ignore_index=True)
    community = all_returns.groupby("trade_date", as_index=False)["daily_return"].mean().sort_values("trade_date")
    community["growth_index"] = (1.0 + community["daily_return"]).cumprod() * 100.0
    community["cumulative_return_pct"] = community["growth_index"] - 100.0
    community["rolling_20d_return_pct"] = ((1.0 + community["daily_return"]).rolling(20).apply(np.prod, raw=True) - 1.0) * 100.0
    community["rolling_20d_vol_pct"] = community["daily_return"].rolling(20).std() * np.sqrt(252) * 100.0
    return community


def ewma_volatility(daily_returns: pd.Series, lam: float = 0.94) -> float:
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    if len(r) < 5:
        return float(r.std()) if len(r) > 1 else 0.01
    variance = float(r.var())
    for x in r.iloc[-60:]:
        variance = lam * variance + (1.0 - lam) * float(x) ** 2
    return float(np.sqrt(max(variance, 1e-10)))


def make_projection_cone(
    daily_returns: pd.Series,
    horizon_days: int = 21,
    n_paths: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    r = pd.to_numeric(daily_returns, errors="coerce").dropna()
    r = r.replace([np.inf, -np.inf], np.nan).dropna().clip(-0.50, 0.50)
    if len(r) < 10:
        return pd.DataFrame()

    mu = float(r.tail(60).mean()) if len(r) >= 20 else float(r.mean())
    sigma = ewma_volatility(r)
    # Guardrails to keep small samples from producing nonsense.
    sigma = min(max(sigma, 0.0025), 0.10)
    mu = min(max(mu, -0.02), 0.02)

    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=mu, scale=sigma, size=(n_paths, horizon_days))
    # A light fat-tail shock layer based on historical residual behavior.
    if len(r) >= 30:
        shock_days = rng.random(size=(n_paths, horizon_days)) < 0.06
        shock_scale = max(float(r.std()) * 2.0, sigma * 1.5)
        shocks = rng.normal(loc=0.0, scale=shock_scale, size=(n_paths, horizon_days))
        draws = np.where(shock_days, draws + shocks, draws)
    draws = np.clip(draws, -0.50, 0.50)
    cumulative = (1.0 + draws).cumprod(axis=1) - 1.0

    rows = []
    for day in range(1, horizon_days + 1):
        vals = cumulative[:, day - 1] * 100.0
        rows.append({
            "projection_day": day,
            "median": np.percentile(vals, 50),
            "p10": np.percentile(vals, 10),
            "p25": np.percentile(vals, 25),
            "p75": np.percentile(vals, 75),
            "p90": np.percentile(vals, 90),
        })
    return pd.DataFrame(rows)


def projection_summary(proj: pd.DataFrame) -> dict:
    if proj.empty:
        return {}
    last = proj.iloc[-1]
    return {
        "median": float(last["median"]),
        "p10": float(last["p10"]),
        "p90": float(last["p90"]),
        "p25": float(last["p25"]),
        "p75": float(last["p75"]),
    }

st.set_page_config(page_title="ALGO Edge Performance History", layout="wide")
init_db()

st.title("ALGO Edge Performance History")
st.caption("Community performance analytics, return projections, and volatility research for systematic traders.")

with st.sidebar:
    st.header("Contribute Performance Data")
    discord_handle = st.text_input("Discord handle", placeholder="@your_handle")
    anonymous = st.checkbox("Upload anonymously", value=False)
    strategy_name = st.text_input("Strategy name", placeholder="Optional")
    account_size = st.number_input("Approx. account size (strongly recommended)", min_value=0.0, value=0.0, step=1000.0)
    notes = st.text_area("Notes / setup description", placeholder="Optional: time period, sizing, symbols, risk rules...")
    show_in_group = st.checkbox("Show my results in the community dashboard", value=True)
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
    assumed_account_size = st.number_input(
        "Return base for uploads missing account size",
        min_value=1000.0,
        value=25000.0,
        step=1000.0,
        help="Used only when an uploader did not provide account size. This converts dollar P/L into approximate returns.",
    )
    st.warning("Before uploading, remove account numbers, brokerage IDs, addresses, or other personal identifiers.")

public_results = load_public_results_with_uploads()
uploads_df = load_uploads()

if public_results.empty:
    st.info("Upload a CSV to begin. The app expects a date column and a daily P/L column.")
    if not uploads_df.empty:
        st.subheader("Private/hidden uploads exist")
        st.write("Some uploads may be hidden from the community dashboard because the uploader unchecked the visibility box.")
    st.stop()

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

if filtered.empty:
    st.warning("Select at least one trader to display the dashboard.")
    st.stop()

community_returns = build_community_daily_returns(filtered, assumed_account_size)
projection = make_projection_cone(community_returns["daily_return"] if not community_returns.empty else pd.Series(dtype=float))
proj_stats = projection_summary(projection)

st.subheader("Projected Community Return — Next 1 Month")
if projection.empty:
    st.info("Projection will appear after at least 10 visible daily return observations are uploaded.")
else:
    pcols = st.columns(4)
    pcols[0].metric("Median projected return", f"{proj_stats['median']:.1f}%")
    pcols[1].metric("Likely range", f"{proj_stats['p25']:.1f}% to {proj_stats['p75']:.1f}%")
    pcols[2].metric("Wide range", f"{proj_stats['p10']:.1f}% to {proj_stats['p90']:.1f}%")
    latest_vol = community_returns["rolling_20d_vol_pct"].dropna()
    pcols[3].metric("Latest realized vol", f"{latest_vol.iloc[-1]:.1f}%" if len(latest_vol) else "n/a")

    fig_proj = px.line(
        projection,
        x="projection_day",
        y=["p10", "p25", "median", "p75", "p90"],
        title="Monte Carlo Projection Cone: Next 21 Trading Days",
        labels={"projection_day": "Trading day", "value": "Projected cumulative return (%)", "variable": "Band"},
    )
    st.plotly_chart(fig_proj, use_container_width=True)
    st.caption("Projection uses historical community daily returns with an EWMA/GARCH-style volatility estimate. It is research output, not a prediction or trading recommendation.")

st.subheader("Community Dashboard")
metric_cols = st.columns(4)
metric_cols[0].metric("Visible traders", filtered["trader_alias"].nunique())
metric_cols[1].metric("Trading days analyzed", f"{len(filtered):,}")
metric_cols[2].metric("Community net P/L", f"${filtered['daily_pl'].sum():,.0f}")
if not community_returns.empty:
    total_return = community_returns["cumulative_return_pct"].iloc[-1]
    metric_cols[3].metric("Community cumulative return", f"{total_return:.1f}%")
else:
    metric_cols[3].metric("Community cumulative return", "n/a")

left, right = st.columns([2, 1])
with right:
    st.markdown("### Trader Summary")
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
    st.markdown("### Normalized Return Curves")
    curve_df = pd.concat([add_return_metrics(g, assumed_account_size) for _, g in filtered.groupby("trader_alias")], ignore_index=True)
    fig = px.line(
        curve_df,
        x="trade_date",
        y="cumulative_return_pct",
        color="trader_alias",
        title="Cumulative Return by Trader",
        labels={"trade_date": "Date", "cumulative_return_pct": "Cumulative return (%)", "trader_alias": "Trader"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Community Return Trend")
if not community_returns.empty:
    fig_comm = px.line(
        community_returns,
        x="trade_date",
        y="cumulative_return_pct",
        title="Community Average Cumulative Return",
        labels={"trade_date": "Date", "cumulative_return_pct": "Cumulative return (%)"},
    )
    st.plotly_chart(fig_comm, use_container_width=True)

st.markdown("### Rolling 20-Day Return")
rolling_df = curve_df.dropna(subset=["rolling_20d_return_pct"])
if rolling_df.empty:
    st.info("Rolling return chart will appear after at least 20 observations per trader.")
else:
    fig_roll = px.line(
        rolling_df,
        x="trade_date",
        y="rolling_20d_return_pct",
        color="trader_alias",
        title="Rolling 20-Day Return by Trader",
        labels={"trade_date": "Date", "rolling_20d_return_pct": "20-day return (%)", "trader_alias": "Trader"},
    )
    st.plotly_chart(fig_roll, use_container_width=True)

st.markdown("### Volatility Trend")
vol_df = curve_df.dropna(subset=["rolling_20d_vol_pct"])
if vol_df.empty:
    st.info("Volatility trend will appear after at least 20 observations per trader.")
else:
    fig_vol = px.line(
        vol_df,
        x="trade_date",
        y="rolling_20d_vol_pct",
        color="trader_alias",
        title="Rolling 20-Day Annualized Realized Volatility",
        labels={"trade_date": "Date", "rolling_20d_vol_pct": "Annualized volatility (%)", "trader_alias": "Trader"},
    )
    st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("### Daily Return Distribution")
fig_hist = px.histogram(
    curve_df,
    x="daily_return",
    color="trader_alias",
    nbins=60,
    marginal="box",
    title="Daily Return Histogram",
    labels={"daily_return": "Daily return", "trader_alias": "Trader"},
)
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("Recent uploads"):
    display_uploads = uploads_df[["uploaded_at", "trader_alias", "strategy_name", "account_size", "row_count", "show_in_group", "original_filename"]].copy()
    st.dataframe(display_uploads, use_container_width=True)

with st.expander("Export visible standardized dataset"):
    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: local Streamlit storage can reset on redeploy. For durable Discord group use, move the database/storage to Supabase next.")
