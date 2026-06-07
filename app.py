
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

TRADING_DAYS_ONE_MONTH = 21
N_SIMULATIONS = 5000


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
    (UPLOAD_DIR / stored_filename).write_bytes(content)

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
    where = "WHERE d.show_in_group = 1" if public_only else ""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT d.*, u.account_size, u.uploaded_at, u.original_filename
            FROM daily_results d
            LEFT JOIN uploads u ON d.upload_id = u.upload_id
            {where}
            """,
            conn,
        )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["daily_pl"] = pd.to_numeric(df["daily_pl"], errors="coerce")
    df["account_size"] = pd.to_numeric(df["account_size"], errors="coerce")
    df["daily_return"] = np.where(df["account_size"] > 0, df["daily_pl"] / df["account_size"], np.nan)
    return df


def load_uploads() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM uploads ORDER BY uploaded_at DESC", conn)


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()
    out["cumulative_pl"] = out["daily_pl"].cumsum()
    out["running_peak"] = out["cumulative_pl"].cummax()
    out["drawdown"] = out["cumulative_pl"] - out["running_peak"]
    if out["daily_return"].notna().any():
        out["growth_index"] = 100 * (1 + out["daily_return"].fillna(0)).cumprod()
        out["cumulative_return_pct"] = out["growth_index"] - 100
    else:
        out["growth_index"] = 100 + out["cumulative_pl"]
        out["cumulative_return_pct"] = out["cumulative_pl"]
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
    stats["first_day"] = grouped["trade_date"].min().dt.date.values
    stats["last_day"] = grouped["trade_date"].max().dt.date.values

    if "daily_return" in df.columns and df["daily_return"].notna().any():
        ret_stats = grouped["daily_return"].apply(lambda s: (1 + s.dropna()).prod() - 1 if s.dropna().size else np.nan)
        stats["total_return"] = stats["trader_alias"].map(ret_stats)
        stats["daily_vol"] = stats["trader_alias"].map(grouped["daily_return"].std())
    return stats.sort_values("total_pl", ascending=False)


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3:
        return float(r.std()) if len(r) > 1 else 0.0
    variance = float(r.var())
    for x in r.iloc[-60:]:
        variance = lam * variance + (1 - lam) * float(x) ** 2
    return float(np.sqrt(max(variance, 0)))


def volatility_regime_label(vol: float, historical_vols: pd.Series) -> str:
    hv = historical_vols.dropna()
    if len(hv) < 10 or not np.isfinite(vol):
        return "Insufficient history"
    p50, p75, p90 = hv.quantile([0.50, 0.75, 0.90])
    if vol < p50:
        return "Low volatility"
    if vol < p75:
        return "Normal volatility"
    if vol < p90:
        return "High volatility"
    return "Extreme volatility"


def build_projection(df: pd.DataFrame, selected_traders: list[str], horizon_days: int = TRADING_DAYS_ONE_MONTH) -> tuple[pd.DataFrame, dict]:
    work = df[df["trader_alias"].isin(selected_traders)].copy()
    use_returns = work["daily_return"].notna().sum() >= 20

    if use_returns:
        daily = (
            work.dropna(subset=["daily_return"])
            .groupby("trade_date")["daily_return"]
            .mean()
            .sort_index()
        )
        units = "return"
        value_suffix = "%"
        multiplier = 100.0
    else:
        daily = work.groupby("trade_date")["daily_pl"].sum().sort_index()
        units = "P/L"
        value_suffix = " dollars"
        multiplier = 1.0

    daily = daily.replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily) < 10:
        return pd.DataFrame(), {"error": "Need at least 10 historical trading days to build a projection."}

    drift = float(daily.tail(63).mean()) if len(daily) >= 20 else float(daily.mean())
    realized_vol = float(daily.tail(min(21, len(daily))).std())
    forecast_vol = ewma_volatility(daily)
    if not np.isfinite(forecast_vol) or forecast_vol <= 0:
        forecast_vol = realized_vol if np.isfinite(realized_vol) else 0.0

    # Conservative blend: recent average drift, with volatility from EWMA/GARCH-style forecast.
    # Drift is clipped so a hot or cold sample does not dominate the one-month projection.
    vol_cap = max(forecast_vol, 1e-9)
    drift = float(np.clip(drift, -0.35 * vol_cap, 0.35 * vol_cap))

    rng = np.random.default_rng(42)
    shocks = rng.normal(loc=drift, scale=forecast_vol, size=(N_SIMULATIONS, horizon_days))

    if use_returns:
        paths = (1 + shocks).cumprod(axis=1) - 1
    else:
        paths = shocks.cumsum(axis=1)

    quantiles = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)
    projection = pd.DataFrame({
        "day": np.arange(1, horizon_days + 1),
        "p10": quantiles[0] * multiplier,
        "p25": quantiles[1] * multiplier,
        "median": quantiles[2] * multiplier,
        "p75": quantiles[3] * multiplier,
        "p90": quantiles[4] * multiplier,
    })

    rolling_vol = daily.rolling(21, min_periods=5).std()
    meta = {
        "units": units,
        "value_suffix": value_suffix,
        "use_returns": use_returns,
        "history_days": len(daily),
        "daily_drift": drift * multiplier,
        "daily_forecast_vol": forecast_vol * multiplier,
        "monthly_median": float(projection["median"].iloc[-1]),
        "monthly_p10": float(projection["p10"].iloc[-1]),
        "monthly_p90": float(projection["p90"].iloc[-1]),
        "regime": volatility_regime_label(forecast_vol, rolling_vol),
    }
    return projection, meta


def plot_projection(projection: pd.DataFrame, meta: dict) -> go.Figure:
    y_title = "Projected return (%)" if meta.get("use_returns") else "Projected P/L ($)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projection["day"], y=projection["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=projection["day"], y=projection["p10"], fill="tonexty", line=dict(width=0),
        name="10%–90% range", hovertemplate="Day %{x}<br>Lower band: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=projection["day"], y=projection["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=projection["day"], y=projection["p25"], fill="tonexty", line=dict(width=0),
        name="25%–75% range", hovertemplate="Day %{x}<br>Lower quartile: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=projection["day"], y=projection["median"], mode="lines", name="Median projection",
        hovertemplate="Day %{x}<br>Median: %{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(
        title="Projected Next 1-Month Return Path",
        xaxis_title="Trading days forward",
        yaxis_title=y_title,
        hovermode="x unified",
        legend_title_text="Projection",
    )
    return fig


def build_volatility_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trader, g in df.groupby("trader_alias"):
        g = g.sort_values("trade_date").copy()
        series = g["daily_return"].dropna() if g["daily_return"].notna().sum() >= 10 else g["daily_pl"].dropna()
        if len(series) < 5:
            continue
        g2 = g.iloc[-len(series):].copy()
        g2["realized_vol_20d"] = series.rolling(20, min_periods=5).std().values
        g2["forecast_vol"] = [ewma_volatility(series.iloc[: i + 1]) for i in range(len(series))]
        g2["trader_alias"] = trader
        rows.append(g2[["trade_date", "trader_alias", "realized_vol_20d", "forecast_vol"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


st.set_page_config(page_title="ALGO Edge Performance History", layout="wide")
init_db()

st.title("ALGO Edge Performance History")
st.caption("Performance analytics, volatility research, and regime monitoring for systematic traders.")

with st.sidebar:
    st.header("Contribute Performance Data")
    discord_handle = st.text_input("Discord handle", placeholder="@your_handle")
    anonymous = st.checkbox("Upload anonymously", value=False)
    strategy_name = st.text_input("Strategy name", placeholder="Optional")
    account_size = st.number_input("Approx. account size for return normalization", min_value=0.0, value=0.0, step=1000.0)
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
    st.stop()

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

st.subheader("Projected 1-Month Return")
projection, projection_meta = build_projection(public_results, selected_traders)
if projection.empty:
    st.warning(projection_meta.get("error", "Projection could not be built from the available history."))
else:
    pcols = st.columns(4)
    if projection_meta["use_returns"]:
        pcols[0].metric("Median 1-month projection", f"{projection_meta['monthly_median']:.2f}%")
        pcols[1].metric("10% downside case", f"{projection_meta['monthly_p10']:.2f}%")
        pcols[2].metric("90% upside case", f"{projection_meta['monthly_p90']:.2f}%")
        pcols[3].metric("Current volatility regime", projection_meta["regime"])
    else:
        pcols[0].metric("Median 1-month projection", f"${projection_meta['monthly_median']:,.0f}")
        pcols[1].metric("10% downside case", f"${projection_meta['monthly_p10']:,.0f}")
        pcols[2].metric("90% upside case", f"${projection_meta['monthly_p90']:,.0f}")
        pcols[3].metric("Current volatility regime", projection_meta["regime"])
    st.plotly_chart(plot_projection(projection, projection_meta), use_container_width=True)
    if not projection_meta["use_returns"]:
        st.info("Projection is shown in dollars because fewer than 20 rows had account size data. Add account size on upload to enable percent return projections.")
    st.caption(
        "Projection uses historical daily results, an EWMA/GARCH-style volatility forecast, and 5,000 Monte Carlo paths over 21 trading days. "
        "It is a risk model, not a promise of future performance."
    )

st.subheader("Community Performance Dashboard")
metric_cols = st.columns(4)
metric_cols[0].metric("Visible traders", filtered["trader_alias"].nunique())
metric_cols[1].metric("Trading days analyzed", f"{len(filtered):,}")
metric_cols[2].metric("Aggregate net P/L", f"${filtered['daily_pl'].sum():,.0f}")
metric_cols[3].metric("Uploads", f"{len(uploads_df):,}")

st.markdown("### Volatility Trend")
vol_df = build_volatility_trend(filtered)
if vol_df.empty:
    st.info("Need more uploaded history to calculate a volatility trend.")
else:
    fig_vol = go.Figure()
    avg_vol = vol_df.groupby("trade_date", as_index=False)[["realized_vol_20d", "forecast_vol"]].mean()
    scale = 100 if filtered["daily_return"].notna().sum() >= 20 else 1
    fig_vol.add_trace(go.Scatter(x=avg_vol["trade_date"], y=avg_vol["realized_vol_20d"] * scale, mode="lines", name="20-day realized volatility"))
    fig_vol.add_trace(go.Scatter(x=avg_vol["trade_date"], y=avg_vol["forecast_vol"] * scale, mode="lines", name="Forecast volatility"))
    fig_vol.update_layout(
        title="Historical Volatility of Returns",
        xaxis_title="Date",
        yaxis_title="Daily volatility (%)" if scale == 100 else "Daily P/L volatility ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

left, right = st.columns([2, 1])
curve_df = pd.concat([add_metrics(g) for _, g in filtered.groupby("trader_alias")], ignore_index=True)

with left:
    st.markdown("### Cumulative Returns")
    if curve_df["daily_return"].notna().any():
        fig = px.line(curve_df, x="trade_date", y="cumulative_return_pct", color="trader_alias", title="Cumulative Return (%)")
        fig.update_yaxes(title="Cumulative return (%)")
    else:
        fig = px.line(curve_df, x="trade_date", y="cumulative_pl", color="trader_alias", title="Cumulative P/L")
        fig.update_yaxes(title="Cumulative P/L ($)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Summary")
    stats = summary_stats(filtered)
    formatters = {
        "total_pl": "${:,.0f}",
        "avg_day": "${:,.0f}",
        "median_day": "${:,.0f}",
        "std_day": "${:,.0f}",
        "best_day": "${:,.0f}",
        "worst_day": "${:,.0f}",
        "win_rate": "{:.1%}",
        "profit_factor": "{:.2f}",
    }
    if "total_return" in stats.columns:
        formatters["total_return"] = "{:.1%}"
        formatters["daily_vol"] = "{:.2%}"
    st.dataframe(stats.style.format(formatters), use_container_width=True)

st.markdown("### Rolling 20-Day Returns")
roll_rows = []
for trader, g in filtered.groupby("trader_alias"):
    g = g.sort_values("trade_date").copy()
    if g["daily_return"].notna().sum() >= 20:
        s = g["daily_return"].fillna(0)
        g["rolling_20d_return"] = ((1 + s).rolling(20).apply(np.prod, raw=True) - 1) * 100
    else:
        g["rolling_20d_return"] = g["daily_pl"].rolling(20).sum()
    roll_rows.append(g)
roll_df = pd.concat(roll_rows, ignore_index=True)
fig_roll = px.line(roll_df, x="trade_date", y="rolling_20d_return", color="trader_alias", title="Rolling 20-Day Return / P&L")
st.plotly_chart(fig_roll, use_container_width=True)

st.markdown("### Daily Return Distribution")
if filtered["daily_return"].notna().sum() >= 20:
    hist_df = filtered.dropna(subset=["daily_return"]).copy()
    hist_df["daily_return_pct"] = hist_df["daily_return"] * 100
    fig_hist = px.histogram(hist_df, x="daily_return_pct", color="trader_alias", nbins=60, marginal="box", title="Daily Return Histogram (%)")
    fig_hist.update_xaxes(title="Daily return (%)")
else:
    fig_hist = px.histogram(filtered, x="daily_pl", color="trader_alias", nbins=60, marginal="box", title="Daily P/L Histogram")
st.plotly_chart(fig_hist, use_container_width=True)

with st.expander("Recent uploads"):
    display_cols = ["uploaded_at", "trader_alias", "row_count", "show_in_group", "original_filename"]
    if "account_size" in uploads_df.columns:
        display_cols.insert(2, "account_size")
    st.dataframe(uploads_df[display_cols], use_container_width=True)

with st.expander("Export visible standardized dataset"):
    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: this version stores data locally in the Streamlit app container. For durable Discord group use, move the database/storage to Supabase next.")
