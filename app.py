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





def add_return_focus_metrics(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """Return-focused metrics using cumulative P/L because account sizes are optional."""
    out = add_metrics(df)
    out["rolling_return_pl"] = out["daily_pl"].rolling(rolling_window, min_periods=max(3, rolling_window // 4)).sum()
    return out


def community_return_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Build a community average cumulative P/L curve by day across selected traders."""
    if df.empty:
        return pd.DataFrame()
    pieces = []
    for trader, g in df.groupby("trader_alias"):
        m = add_metrics(g)
        pieces.append(m[["trade_date", "trader_alias", "cumulative_pl"]])
    all_curves = pd.concat(pieces, ignore_index=True)
    return all_curves.groupby("trade_date", as_index=False)["cumulative_pl"].mean().rename(columns={"cumulative_pl": "community_avg_cumulative_pl"})

def calculate_volatility_trend(df: pd.DataFrame, window: int = 20, annualization: int = 252) -> pd.DataFrame:
    """
    Build a visible volatility trend from daily P/L.

    If account_size exists in the upload metadata later, this can be converted to true returns.
    For the current no-login MVP, we standardize by each trader's own daily P/L volatility so
    the chart compares volatility regimes rather than dollar size.
    """
    if df.empty:
        return pd.DataFrame()

    pieces = []
    for trader, g in df.sort_values('trade_date').groupby('trader_alias'):
        out = g.copy()
        scale = out['daily_pl'].abs().median()
        if not np.isfinite(scale) or scale == 0:
            scale = out['daily_pl'].std()
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        out['standardized_return'] = out['daily_pl'] / scale
        out['rolling_realized_vol'] = out['standardized_return'].rolling(window, min_periods=max(5, window // 4)).std() * np.sqrt(annualization)

        # GARCH-style forecast using an EWMA variance recursion.
        # This is intentionally dependency-light for Streamlit Cloud reliability.
        lam = 0.94
        values = out['standardized_return'].fillna(0).to_numpy(dtype=float)
        if len(values) == 0:
            continue
        seed = np.nanvar(values[: min(len(values), window)])
        if not np.isfinite(seed) or seed <= 0:
            seed = np.nanvar(values) if np.isfinite(np.nanvar(values)) else 0.0
        var = max(seed, 1e-8)
        forecast = []
        for r in values:
            var = lam * var + (1 - lam) * (r ** 2)
            forecast.append(np.sqrt(var * annualization))
        out['garch_style_forecast_vol'] = forecast
        pieces.append(out)

    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)




def build_projection_source(df: pd.DataFrame, target: str) -> pd.Series:
    """Return a daily P/L series for either the community average or one trader."""
    if df.empty:
        return pd.Series(dtype=float)
    ordered = df.sort_values("trade_date").copy()
    if target == "Community average":
        # Average each trader's daily P/L by date so one very active uploader does not dominate.
        daily_by_trader = ordered.groupby(["trade_date", "trader_alias"], as_index=False)["daily_pl"].sum()
        series = daily_by_trader.groupby("trade_date")["daily_pl"].mean().sort_index()
    else:
        series = ordered[ordered["trader_alias"] == target].groupby("trade_date")["daily_pl"].sum().sort_index()
    return pd.to_numeric(series, errors="coerce").dropna()


def ewma_daily_vol(values: np.ndarray, lam: float = 0.94) -> float:
    """Dependency-light GARCH-style volatility estimate from daily P/L changes."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    centered = values - np.nanmean(values)
    seed = np.nanvar(centered, ddof=1) if len(centered) > 2 else np.nanvar(centered)
    if not np.isfinite(seed) or seed <= 0:
        return float("nan")
    var = seed
    for r in centered:
        var = lam * var + (1 - lam) * (r ** 2)
    return float(np.sqrt(max(var, 1e-12)))


def monte_carlo_one_month_projection(
    daily_pl: pd.Series,
    horizon_days: int = 21,
    n_sims: int = 5000,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Project cumulative P/L over the next trading month using historical mean and EWMA volatility."""
    values = pd.to_numeric(daily_pl, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 10:
        return pd.DataFrame(), {"error": "Need at least 10 trading days for a projection."}

    mu = float(np.nanmean(values))
    hist_vol = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    forecast_vol = ewma_daily_vol(values)
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        forecast_vol = hist_vol
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        return pd.DataFrame(), {"error": "Daily P/L volatility is zero or unavailable."}

    # Blend long-run historical volatility with recent EWMA volatility for stability.
    daily_vol = float(0.50 * hist_vol + 0.50 * forecast_vol) if np.isfinite(hist_vol) and hist_vol > 0 else float(forecast_vol)

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=daily_vol, size=(n_sims, horizon_days))
    paths = shocks.cumsum(axis=1)

    rows = []
    for i in range(horizon_days):
        vals = paths[:, i]
        rows.append({
            "trading_day": i + 1,
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
        })
    meta = {
        "observations": int(len(values)),
        "avg_daily_pl": mu,
        "historical_daily_vol": hist_vol,
        "forecast_daily_vol": forecast_vol,
        "blended_daily_vol": daily_vol,
        "projected_median": rows[-1]["median"],
        "projected_p10": rows[-1]["p10"],
        "projected_p90": rows[-1]["p90"],
    }
    return pd.DataFrame(rows), meta


def plot_projection(proj: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proj["trading_day"], y=proj["p90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=proj["trading_day"], y=proj["p10"], mode="lines", fill="tonexty", line=dict(width=0),
        name="10–90% range", hovertemplate="Day %{x}<br>10th pct: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=proj["trading_day"], y=proj["p75"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=proj["trading_day"], y=proj["p25"], mode="lines", fill="tonexty", line=dict(width=0),
        name="25–75% range", hovertemplate="Day %{x}<br>25th pct: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=proj["trading_day"], y=proj["median"], mode="lines", name="Median projection",
        hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Trading days ahead",
        yaxis_title="Projected cumulative P/L",
        hovermode="x unified",
    )
    return fig


def volatility_regime_label(value: float, q_low: float, q_high: float) -> str:
    if not np.isfinite(value):
        return 'Insufficient data'
    if value >= q_high:
        return 'High volatility'
    if value <= q_low:
        return 'Low volatility'
    return 'Normal volatility'

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


st.set_page_config(page_title="ALGO Edge Performance History", layout="wide")
init_db()

st.title("ALGO Edge Performance History")
st.caption("Performance analytics, volatility research, and regime monitoring for systematic traders.")

with st.sidebar:
    st.header("Upload Trading History")
    discord_handle = st.text_input("Discord handle", placeholder="@your_handle")
    anonymous = st.checkbox("Upload anonymously", value=False)
    strategy_name = st.text_input("Strategy name", placeholder="0DTE Range / Power Hour / etc.")
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

st.subheader("Community Performance Dashboard")
summary = summary_stats(public_results)

metric_cols = st.columns(4)
metric_cols[0].metric("Visible traders", public_results["trader_alias"].nunique())
metric_cols[1].metric("Total trading days", f"{len(public_results):,}")
metric_cols[2].metric("Group net P/L", f"${public_results['daily_pl'].sum():,.0f}")
metric_cols[3].metric("Uploads", f"{len(uploads_df):,}")

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

st.subheader("Projected 1-Month Returns")
st.caption("Front-page projection using historical daily P/L behavior and a GARCH-style EWMA volatility estimate. This is a scenario cone, not a prediction or trading recommendation.")

projection_targets = ["Community average"] + selected_traders
projection_target = st.selectbox("Projection target", projection_targets, index=0)
source_series = build_projection_source(filtered, projection_target)
proj_df, proj_meta = monte_carlo_one_month_projection(source_series)

if proj_df.empty:
    st.info(proj_meta.get("error", "Upload more data to generate a 1-month projection."))
else:
    pcols = st.columns(4)
    pcols[0].metric("Days used", f"{proj_meta['observations']:,}")
    pcols[1].metric("Avg daily P/L", f"${proj_meta['avg_daily_pl']:,.0f}")
    pcols[2].metric("Forecast daily vol", f"${proj_meta['forecast_daily_vol']:,.0f}")
    pcols[3].metric("1-month median", f"${proj_meta['projected_median']:,.0f}")

    st.plotly_chart(
        plot_projection(proj_df, f"Projected Next 21 Trading Days: {projection_target}"),
        use_container_width=True,
    )

    with st.expander("Projection assumptions"):
        st.write(
            "The projection simulates 5,000 one-month paths using the historical average daily P/L "
            "and a volatility estimate that blends full-history volatility with recent EWMA/GARCH-style volatility. "
            "The shaded bands show simulated percentile ranges."
        )
        st.dataframe(
            pd.DataFrame([proj_meta]).style.format({
                "avg_daily_pl": "${:,.0f}",
                "historical_daily_vol": "${:,.0f}",
                "forecast_daily_vol": "${:,.0f}",
                "blended_daily_vol": "${:,.0f}",
                "projected_median": "${:,.0f}",
                "projected_p10": "${:,.0f}",
                "projected_p90": "${:,.0f}",
            }),
            use_container_width=True,
        )

st.subheader("Volatility Trend")
st.caption("This section shows the volatility trend directly: rolling realized volatility and a GARCH-style EWMA volatility forecast from standardized daily P/L. It is designed for relative regime detection across traders, not brokerage-grade risk reporting yet.")

vol_window = st.slider("Rolling volatility window", min_value=5, max_value=60, value=20, step=5)
vol_df = calculate_volatility_trend(filtered, window=vol_window)

if vol_df.empty or vol_df["rolling_realized_vol"].dropna().empty:
    st.info("Upload more trading days to show a volatility trend. The chart needs several observations before rolling volatility is meaningful.")
else:
    latest_vol = (
        vol_df.sort_values("trade_date")
        .groupby("trader_alias")
        .tail(1)[["trader_alias", "rolling_realized_vol", "garch_style_forecast_vol"]]
        .copy()
    )
    q_low = vol_df["garch_style_forecast_vol"].quantile(0.25)
    q_high = vol_df["garch_style_forecast_vol"].quantile(0.75)
    latest_vol["regime"] = latest_vol["garch_style_forecast_vol"].apply(lambda x: volatility_regime_label(x, q_low, q_high))

    vcols = st.columns(3)
    vcols[0].metric("Latest realized vol", f"{latest_vol['rolling_realized_vol'].mean():.2f}x")
    vcols[1].metric("Latest forecast vol", f"{latest_vol['garch_style_forecast_vol'].mean():.2f}x")
    vcols[2].metric("Group regime", volatility_regime_label(latest_vol['garch_style_forecast_vol'].mean(), q_low, q_high))

    vol_long = vol_df.melt(
        id_vars=["trade_date", "trader_alias"],
        value_vars=["rolling_realized_vol", "garch_style_forecast_vol"],
        var_name="volatility_measure",
        value_name="volatility",
    ).dropna(subset=["volatility"])
    vol_long["volatility_measure"] = vol_long["volatility_measure"].replace({
        "rolling_realized_vol": "Rolling realized volatility",
        "garch_style_forecast_vol": "GARCH-style forecast volatility",
    })
    fig_vol = px.line(
        vol_long,
        x="trade_date",
        y="volatility",
        color="trader_alias",
        line_dash="volatility_measure",
        title="Volatility Trend: Realized vs Forecast",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    with st.expander("Latest volatility regime by trader"):
        st.dataframe(
            latest_vol.style.format({
                "rolling_realized_vol": "{:.2f}x",
                "garch_style_forecast_vol": "{:.2f}x",
            }),
            use_container_width=True,
        )


st.subheader("Return Performance")
st.caption("This view intentionally focuses on return history rather than strategy labels. Because contributors may use different account sizes and sizing rules, the MVP uses cumulative daily P/L and rolling daily P/L. Account-normalized returns can be added once uploads consistently include account size or beginning equity.")

curve_df = pd.concat([add_return_focus_metrics(g) for _, g in filtered.groupby("trader_alias")], ignore_index=True)
community_curve = community_return_curve(filtered)

left, right = st.columns([2, 1])
with left:
    st.markdown("### Cumulative Return History")
    fig = px.line(
        curve_df,
        x="trade_date",
        y="cumulative_pl",
        color="trader_alias",
        title="Cumulative Daily P/L by Trader",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not community_curve.empty:
        fig_group = px.line(
            community_curve,
            x="trade_date",
            y="community_avg_cumulative_pl",
            title="Community Average Cumulative Daily P/L",
        )
        st.plotly_chart(fig_group, use_container_width=True)

with right:
    st.markdown("### Return Summary")
    stats = summary_stats(filtered)
    display_cols = [
        "trader_alias", "days", "total_pl", "avg_day", "median_day", "std_day",
        "best_day", "worst_day", "win_rate", "profit_factor", "first_day", "last_day"
    ]
    st.dataframe(
        stats[display_cols].style.format({
            "total_pl": "${:,.0f}",
            "avg_day": "${:,.0f}",
            "median_day": "${:,.0f}",
            "std_day": "${:,.0f}",
            "best_day": "${:,.0f}",
            "worst_day": "${:,.0f}",
            "win_rate": "{:.1%}",
            "profit_factor": "{:.2f}",
        }),
        use_container_width=True,
    )

st.markdown("### Rolling Return Trend")
rolling_window = st.slider("Rolling return window", min_value=5, max_value=60, value=20, step=5)
rolling_df = pd.concat([add_return_focus_metrics(g, rolling_window) for _, g in filtered.groupby("trader_alias")], ignore_index=True)
fig_roll = px.line(
    rolling_df.dropna(subset=["rolling_return_pl"]),
    x="trade_date",
    y="rolling_return_pl",
    color="trader_alias",
    title=f"Rolling {rolling_window}-Trading-Day P/L",
)
st.plotly_chart(fig_roll, use_container_width=True)

st.markdown("### Daily Return Distribution")
fig_hist = px.histogram(
    filtered,
    x="daily_pl",
    color="trader_alias",
    nbins=60,
    marginal="box",
    title="Daily P/L Distribution",
)
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("Recent uploads"):
    display_uploads = uploads_df[["uploaded_at", "trader_alias", "strategy_name", "row_count", "show_in_group", "original_filename"]].copy()
    st.dataframe(display_uploads, use_container_width=True)

with st.expander("Export visible standardized dataset"):
    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: this version stores data locally in the Streamlit app container. For durable Discord group use, move the database/storage to Supabase next.")
