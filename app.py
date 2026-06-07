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

try:
    from scipy.optimize import minimize
except Exception:  # Streamlit can still run basic dashboards if scipy install fails.
    minimize = None

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
    stats["max_drawdown"] = grouped.apply(lambda g: add_metrics(g)["drawdown"].min()).values
    stats["first_day"] = grouped["trade_date"].min().dt.date.values
    stats["last_day"] = grouped["trade_date"].max().dt.date.values
    return stats.sort_values("total_pl", ascending=False)


def choose_scale(df: pd.DataFrame, account_size_override: float | None = None) -> float:
    if account_size_override and account_size_override > 0:
        return float(account_size_override)
    uploads = load_uploads()
    aliases = df["trader_alias"].dropna().unique().tolist()
    acct = uploads[uploads["trader_alias"].isin(aliases)]["account_size"].dropna()
    if len(acct) > 0 and acct.mean() > 0:
        return float(acct.mean())
    abs95 = df["daily_pl"].abs().quantile(0.95)
    return max(float(abs95 * 20), 25_000.0)


def build_volatility_series(df: pd.DataFrame, account_size_override: float | None = None) -> pd.DataFrame:
    """Aggregate selected P/L and convert it to daily returns for volatility modeling."""
    daily = df.groupby("trade_date", as_index=False)["daily_pl"].sum().sort_values("trade_date")
    scale = choose_scale(df, account_size_override)
    daily["return"] = daily["daily_pl"] / scale
    daily["return_pct"] = daily["return"] * 100
    daily["abs_return_pct"] = daily["return_pct"].abs()
    daily["realized_vol_5d"] = daily["return"].rolling(5).std() * np.sqrt(252) * 100
    daily["realized_vol_20d"] = daily["return"].rolling(20).std() * np.sqrt(252) * 100
    return daily


def garch_neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    var = np.empty_like(r)
    var0 = max(np.var(r), 1e-8)
    var[0] = var0
    for i in range(1, len(r)):
        var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]
        if var[i] <= 0 or not np.isfinite(var[i]):
            return 1e12
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + (r ** 2 / var)))


def fit_garch_11(returns: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Fit a simple zero-mean GARCH(1,1). Falls back to stable defaults when sample is small."""
    r = returns.dropna().astype(float).values
    r = r[np.isfinite(r)]
    if len(r) < 15:
        params = {"omega": np.nan, "alpha": 0.10, "beta": 0.85, "persistence": 0.95, "method": "default: need 15+ observations"}
    elif minimize is None:
        params = {"omega": max(np.var(r) * 0.05, 1e-8), "alpha": 0.10, "beta": 0.85, "persistence": 0.95, "method": "default: scipy unavailable"}
    else:
        var = max(np.var(r), 1e-8)
        x0 = np.array([var * 0.05, 0.10, 0.85])
        bounds = [(1e-10, var * 10 + 1e-6), (0.001, 0.40), (0.001, 0.98)]
        constraints = ({"type": "ineq", "fun": lambda x: 0.999 - x[1] - x[2]},)
        result = minimize(garch_neg_loglik, x0, args=(r,), method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500})
        if result.success:
            omega, alpha, beta = result.x
            params = {"omega": float(omega), "alpha": float(alpha), "beta": float(beta), "persistence": float(alpha + beta), "method": "maximum likelihood"}
        else:
            params = {"omega": var * 0.05, "alpha": 0.10, "beta": 0.85, "persistence": 0.95, "method": "fallback: optimizer failed"}

    omega = params["omega"] if np.isfinite(params["omega"]) else max(np.var(r) * 0.05 if len(r) else 1e-6, 1e-8)
    alpha, beta = params["alpha"], params["beta"]
    var_path = np.empty(len(r))
    var_path[0] = max(np.var(r), 1e-8) if len(r) else 1e-8
    for i in range(1, len(r)):
        var_path[i] = omega + alpha * r[i - 1] ** 2 + beta * var_path[i - 1]
    model = pd.DataFrame({"return": r, "garch_daily_vol": np.sqrt(var_path), "garch_annual_vol_pct": np.sqrt(var_path) * np.sqrt(252) * 100})
    if len(r):
        next_var = omega + alpha * r[-1] ** 2 + beta * var_path[-1]
        params["next_day_vol_pct"] = float(np.sqrt(next_var) * 100)
        params["next_annual_vol_pct"] = float(np.sqrt(next_var) * np.sqrt(252) * 100)
    else:
        params["next_day_vol_pct"] = np.nan
        params["next_annual_vol_pct"] = np.nan
    return model, params


def classify_regime(vol: float, q_low: float, q_high: float) -> str:
    if not np.isfinite(vol):
        return "Unknown"
    if vol <= q_low:
        return "Low volatility"
    if vol >= q_high:
        return "High volatility"
    return "Normal volatility"


st.set_page_config(page_title="Discord Trading Research Repository", layout="wide")
init_db()

st.title("Discord Trading Research Repository")
st.caption("Open upload portal for TradeSteward-style CSVs with shared performance and GARCH-style volatility analytics.")

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
    st.stop()

traders = sorted(public_results["trader_alias"].dropna().unique())
selected_traders = st.multiselect("Show traders", traders, default=traders)
filtered = public_results[public_results["trader_alias"].isin(selected_traders)].copy()

if filtered.empty:
    st.warning("Select at least one trader.")
    st.stop()

tab_group, tab_vol, tab_strategy, tab_export = st.tabs(["Group Dashboard", "GARCH Volatility", "Strategy Breakdown", "Data / Export"])

with tab_group:
    st.subheader("Group Dashboard")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Visible traders", filtered["trader_alias"].nunique())
    metric_cols[1].metric("Trading rows", f"{len(filtered):,}")
    metric_cols[2].metric("Net P/L", f"${filtered['daily_pl'].sum():,.0f}")
    metric_cols[3].metric("Uploads", f"{len(uploads_df):,}")

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
        st.markdown("### Equity Curves")
        curve_df = pd.concat([add_metrics(g) for _, g in filtered.groupby("trader_alias")], ignore_index=True)
        fig = px.line(curve_df, x="trade_date", y="cumulative_pl", color="trader_alias", title="Cumulative P/L")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Drawdowns")
    fig_dd = px.line(curve_df, x="trade_date", y="drawdown", color="trader_alias", title="Drawdown from Running Peak")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("### Daily P/L Distribution")
    fig_hist = px.histogram(filtered, x="daily_pl", color="trader_alias", nbins=60, marginal="box", title="Daily P/L Histogram")
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_vol:
    st.subheader("GARCH Volatility Dashboard")
    st.caption("Models selected traders as one aggregate book. P/L is converted to return using account size when available, otherwise a conservative inferred capital base.")

    acct_override = st.number_input("Override capital base for return conversion", min_value=0.0, value=0.0, step=5000.0, help="Use this if uploaded account sizes are missing or inconsistent. Leave at 0 to infer.")
    daily = build_volatility_series(filtered, acct_override if acct_override > 0 else None)
    model, params = fit_garch_11(daily["return"])
    model["trade_date"] = daily["trade_date"].values[: len(model)]
    model["daily_pl"] = daily["daily_pl"].values[: len(model)]
    model["return_pct"] = daily["return_pct"].values[: len(model)]
    model["realized_vol_5d"] = daily["realized_vol_5d"].values[: len(model)]
    model["realized_vol_20d"] = daily["realized_vol_20d"].values[: len(model)]

    q_low = model["garch_annual_vol_pct"].quantile(0.25)
    q_high = model["garch_annual_vol_pct"].quantile(0.75)
    current_vol = model["garch_annual_vol_pct"].iloc[-1] if len(model) else np.nan
    current_regime = classify_regime(current_vol, q_low, q_high)

    cols = st.columns(5)
    cols[0].metric("Current regime", current_regime)
    cols[1].metric("GARCH annual vol", f"{current_vol:,.1f}%" if np.isfinite(current_vol) else "n/a")
    cols[2].metric("Next-day vol forecast", f"{params['next_day_vol_pct']:,.2f}%" if np.isfinite(params["next_day_vol_pct"]) else "n/a")
    cols[3].metric("Persistence α+β", f"{params['persistence']:.3f}" if np.isfinite(params["persistence"]) else "n/a")
    cols[4].metric("Model method", params["method"])

    st.markdown("### Volatility Forecast")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=model["trade_date"], y=model["garch_annual_vol_pct"], mode="lines", name="GARCH annualized vol"))
    fig_vol.add_trace(go.Scatter(x=model["trade_date"], y=model["realized_vol_20d"], mode="lines", name="20-day realized vol"))
    fig_vol.add_trace(go.Scatter(x=model["trade_date"], y=model["realized_vol_5d"], mode="lines", name="5-day realized vol"))
    fig_vol.add_hrect(y0=0, y1=q_low, opacity=0.08, line_width=0, annotation_text="Low", annotation_position="top left")
    fig_vol.add_hrect(y0=q_high, y1=max(model["garch_annual_vol_pct"].max() * 1.05, q_high), opacity=0.08, line_width=0, annotation_text="High", annotation_position="top left")
    fig_vol.update_layout(yaxis_title="Annualized volatility (%)", xaxis_title="Date", hovermode="x unified")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("### Returns vs Volatility")
    fig_combo = go.Figure()
    fig_combo.add_trace(go.Bar(x=model["trade_date"], y=model["return_pct"], name="Daily return %", yaxis="y"))
    fig_combo.add_trace(go.Scatter(x=model["trade_date"], y=model["garch_annual_vol_pct"], mode="lines", name="GARCH annual vol %", yaxis="y2"))
    fig_combo.update_layout(
        yaxis=dict(title="Daily return (%)"),
        yaxis2=dict(title="Annualized vol (%)", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    st.markdown("### Regime Performance")
    model["regime"] = model["garch_annual_vol_pct"].apply(lambda v: classify_regime(v, q_low, q_high))
    regime_stats = model.groupby("regime").agg(
        days=("return", "count"),
        avg_return_pct=("return_pct", "mean"),
        total_pl=("daily_pl", "sum"),
        worst_day_pct=("return_pct", "min"),
        best_day_pct=("return_pct", "max"),
    ).reset_index()
    st.dataframe(regime_stats.style.format({"avg_return_pct": "{:.2f}%", "total_pl": "${:,.0f}", "worst_day_pct": "{:.2f}%", "best_day_pct": "{:.2f}%"}), use_container_width=True)

    with st.expander("Model parameters"):
        st.write({k: v for k, v in params.items() if k in ["omega", "alpha", "beta", "persistence", "method"]})
        st.write("Interpretation: higher persistence means volatility shocks decay more slowly. For short-vol / premium strategies, rising forecast volatility can be treated as a sizing or caution signal rather than a directional prediction.")

with tab_strategy:
    st.subheader("Strategy Breakdown")
    strategy = filtered.groupby(["trader_alias", "strategy"], dropna=False)["daily_pl"].agg(["count", "sum", "mean", "min", "max"]).reset_index()
    st.dataframe(
        strategy.style.format({"sum": "${:,.0f}", "mean": "${:,.0f}", "min": "${:,.0f}", "max": "${:,.0f}"}),
        use_container_width=True,
    )

with tab_export:
    st.subheader("Data / Export")
    with st.expander("Recent uploads", expanded=True):
        display_uploads = uploads_df[["uploaded_at", "trader_alias", "strategy_name", "row_count", "show_in_group", "original_filename"]].copy()
        st.dataframe(display_uploads, use_container_width=True)

    csv = public_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download standardized group CSV", csv, "standardized_group_trading_history.csv", "text/csv")

st.caption("MVP note: this version stores data locally in the Streamlit app container. For durable Discord group use, move the database/storage to Supabase next.")
