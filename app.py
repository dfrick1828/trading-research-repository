
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import base64

def image_to_base64(path):
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")

st.set_page_config(
    page_title="Water Right Outlook",
    page_icon="💧",
    layout="wide",
)

REGIME_ORDER = [
    "Free River",
    "Mild Administration",
    "Normal Administration",
    "Senior Administration",
    "Exceptional Administration",
]

REGIME_COLORS = {
    "Free River": "#5ec9df",
    "Mild Administration": "#f1d36b",
    "Normal Administration": "#f2a34a",
    "Senior Administration": "#e45f56",
    "Exceptional Administration": "#7c4cc2",
}

PUBLIC_LABELS = {
    "Free River": "Free River",
    "Mild Administration": "Light Administration",
    "Normal Administration": "Active Administration",
    "Senior Administration": "Restrictive Administration",
    "Exceptional Administration": "Senior Calls Dominating",
}

PUBLIC_SHORT = {
    "Free River": "Free River",
    "Mild Administration": "Light Administration",
    "Normal Administration": "Active Administration",
    "Senior Administration": "Restrictive Administration",
    "Exceptional Administration": "Senior Calls Dominating",
}

PUBLIC_EXPLAIN = {
    "Free River": "Water-right calls are not currently controlling the basin in a meaningful way.",
    "Mild Administration": "Calls are active, but administration is relatively light compared with historic stress periods.",
    "Normal Administration": "Calls are active and water rights are being administered by priority.",
    "Senior Administration": "Senior priorities are controlling the river system and junior water rights may experience limitations.",
    "Exceptional Administration": "Senior calls are dominating the system and administration is unusually restrictive for this time of year.",
}


# -----------------------------
# CDSS live API integration
# -----------------------------
CDSS_BASE_URL = "https://dwr.state.co.us/Rest/GET/api/v2"
INCLUDED_WDS = {1, 3, 69}
EXCLUDED_WDS = {2, 4, 5, 6, 7, 8, 9}

def get_cdss_api_key():
    """Optional: set CDSS_API_KEY in Streamlit secrets or environment."""
    try:
        key = st.secrets.get("CDSS_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("CDSS_API_KEY", "")

def parse_cdss_date(value):
    """Parse CDSS date strings safely one value at a time."""
    if value is None or pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if s in {"", "None", "nan", "NaN", "NaT", "<NA>"}:
        return pd.NaT
    try:
        # CDSS often returns ISO strings with offsets. Parse as UTC, then strip tz.
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(ts):
            return pd.NaT
        return ts.tz_convert(None)
    except Exception:
        try:
            cleaned = s.replace("T", " ").replace("Z", "")
            if len(cleaned) > 6 and cleaned[-6] in ["+", "-"] and cleaned[-3] == ":":
                cleaned = cleaned[:-6]
            return pd.to_datetime(cleaned, errors="coerce")
        except Exception:
            return pd.NaT


def format_call_date_mmddyyyy(value):
    """Format CDSS priority/call dates as MM/DD/YYYY for public display."""
    try:
        ts = parse_cdss_date(value) if "parse_cdss_date" in globals() else pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return "—"
        return pd.Timestamp(ts).strftime("%m/%d/%Y")
    except Exception:
        return "—"

def cdss_get_json(endpoint, params=None):
    params = dict(params or {})
    params.setdefault("format", "json")
    params.setdefault("pageSize", "500000")
    headers = {}
    api_key = get_cdss_api_key()
    if api_key:
        headers["ApiKey"] = api_key
    url = f"{CDSS_BASE_URL}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params, headers=headers, timeout=35)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "ResultList" in data:
        rows = data["ResultList"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    return pd.DataFrame(rows), r.url

@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_live_active_calls():
    """
    Pull active administrative calls from CDSS/DWR.
    Filters to WD1, WD3, WD69 and excludes WDs 2,4,5,6,7,8,9.
    """
    df, url = cdss_get_json("administrativecalls/active", {"division": 1})
    if df.empty:
        return df, url, "No active calls returned from CDSS."

    # Normalize camelCase API names to the historical column names used by the model.
    rename = {
        "dateTimeSet": "Date Time Set",
        "dateTimeReleased": "Date Time Released",
        "waterSourceName": "Water Source",
        "locationWdid": "Location WDID",
        "locationStructureName": "Location Structure Name",
        "priorityWdid": "Call Priority WDID",
        "priorityStructureName": "Priority Structure Name",
        "priorityAdminNumber": "Priority Admin No",
        "priorityDate": "Priority Date",
        "priorityNumber": "Priority No",
        "boundingWdid": "Bounding WDID",
        "boundingStructureName": "Bounding Structure Name",
        "setComments": "Set Comments",
        "releaseComment": "Release Comments",
        "modified": "Modified",
        "moreInformation": "More_Information",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "Location WDID" not in df.columns:
        return pd.DataFrame(), url, "CDSS response did not include Location WDID."

    df["Location WDID"] = pd.to_numeric(df["Location WDID"], errors="coerce")
    df["wd"] = (df["Location WDID"] // 100000).astype("Int64")
    df = df[df["wd"].isin(INCLUDED_WDS) & ~df["wd"].isin(EXCLUDED_WDS)].copy()
    return df, url, "Live CDSS active calls loaded."

def classify_regime_from_priority(set_dt, priority_dt):
    if pd.isna(set_dt) or pd.isna(priority_dt):
        return "Free River", 0, 0

    py = int(pd.Timestamp(priority_dt).year)
    july15 = pd.Timestamp(year=pd.Timestamp(set_dt).year, month=7, day=15)

    if py >= 1871:
        return "Mild Administration", 1, 25
    if 1867 <= py <= 1870:
        return "Normal Administration", 2, 50
    if 1863 <= py <= 1866:
        return "Senior Administration", 3, 75

    # 1862 and earlier
    if pd.Timestamp(set_dt).normalize() < july15:
        return "Exceptional Administration", 4, 100
    return "Senior Administration", 3, 75

def build_live_current_state(live_calls):
    """Return current basin state from the active Cache la Poudre call display rule."""
    if live_calls is None or live_calls.empty:
        return {
            "regime": "Free River",
            "severity": 0,
            "score": 0,
            "historical_percentile": 0,
            "controlling_wd": None,
            "controlling_priority_date": "—",
            "controlling_priority_structure": "—",
            "controlling_water_source": "—",
            "data_source": "Live CDSS active calls",
        }

    df = live_calls.copy()
    df["set_dt"] = df.get("Date Time Set", pd.Series(index=df.index, dtype="object")).apply(parse_cdss_date)
    df["priority_dt"] = df.get("Priority Date", pd.Series(index=df.index, dtype="object")).apply(parse_cdss_date)

    classified = df.apply(
        lambda r: classify_regime_from_priority(r["set_dt"], r["priority_dt"]),
        axis=1,
        result_type="expand",
    )
    classified.columns = ["regime", "severity", "score"]
    df = pd.concat([df, classified], axis=1)

    # Public-facing current call: active Cache la Poudre call.
    # Do not report the most senior call basin-wide. For the public display,
    # report the current Poudre call record from WD3, preferring rows whose
    # water source or structure fields contain "POUDRE", then selecting the
    # most recently set active call.
    df["Priority Admin No"] = pd.to_numeric(df.get("Priority Admin No"), errors="coerce")

    search_cols = [c for c in ["Water Source", "Location Structure Name", "Priority Structure Name"] if c in df.columns]
    if search_cols:
        poudre_mask = False
        for col in search_cols:
            poudre_mask = poudre_mask | df[col].astype(str).str.upper().str.contains("POUDRE", na=False)
    else:
        poudre_mask = False

    report_df = df[(df["wd"] == 3) & poudre_mask].copy()
    if report_df.empty:
        report_df = df[df["wd"] == 3].copy()
    if report_df.empty:
        report_df = df.copy()

    report_df = report_df.sort_values(["set_dt", "Priority Admin No"], ascending=[False, True], na_position="last")
    top = report_df.iloc[0]

    return {
        "regime": top["regime"],
        "severity": int(top["severity"]),
        "score": int(top["score"]),
        "historical_percentile": None,
        "controlling_wd": int(top["wd"]) if pd.notna(top.get("wd")) else None,
        "controlling_priority_date": format_call_date_mmddyyyy(top.get("Priority Date", "—")),
        "controlling_priority_structure": str(top.get("Priority Structure Name", "—")),
        "controlling_water_source": str(top.get("Water Source", "—")),
        "data_source": "Live CDSS active calls",
    }

def build_forecast_from_state(model_df, clf, le, state, selected_date):
    """Create a model feature row using historical analog features plus live regime state."""
    # Use same day-of-year row from latest/selected date as feature template.
    template = get_model_row_for_date(model_df, selected_date)
    row = template.copy()
    row.loc[row.index[0], "regime"] = state["regime"]
    row.loc[row.index[0], "severity"] = state["severity"]
    row.loc[row.index[0], "score"] = state["score"]
    row.loc[row.index[0], "historical_percentile"] = (model_df["score"].dropna() <= state["score"]).mean() * 100
    X = row[FEATURES].fillna(model_df[FEATURES].median(numeric_only=True))
    prob = clf.predict_proba(X)[0]
    raw = pd.Series(prob, index=le.inverse_transform(np.arange(len(prob)))).reindex(REGIME_ORDER, fill_value=0)
    adjusted = apply_expert_probability_overlay(raw, state["regime"], selected_date).sort_values(ascending=False)
    return row, raw, adjusted


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% 0%, rgba(94,201,223,.15), transparent 28%),
    radial-gradient(circle at 83% 2%, rgba(124,76,194,.14), transparent 32%),
    linear-gradient(180deg, #061017, #0c1117 68%, #070b10);
  color: #edf4f7;
  font-family: Inter, sans-serif;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1220px; }

.hero {
  min-height: 440px;
  border: 1px solid rgba(255,255,255,.13);
  border-radius: 34px;
  padding: 42px;
  background:
    linear-gradient(90deg, rgba(4,12,18,.97), rgba(4,12,18,.74), rgba(4,12,18,.28)),
    url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=80');
  background-size: cover;
  background-position: center;
  box-shadow: 0 35px 110px rgba(0,0,0,.42);
  margin-bottom: 26px;
  position: relative;
}
.eyebrow {
  color: #5ec9df;
  text-transform: uppercase;
  letter-spacing: .20em;
  font-weight: 900;
  font-size: 12px;
}
.hero h1 {
  font-size: 66px;
  line-height: .92;
  letter-spacing: -.065em;
  margin: 18px 0 18px;
  color: #f5fafc;
  max-width: 920px;
}
.hero p {
  max-width: 760px;
  font-size: 22px;
  line-height: 1.45;
  color: #dbe8ee;
}
.hero-footer {
  position:absolute;
  left:42px;
  right:42px;
  bottom:34px;
  display:flex;
  justify-content:space-between;
  gap:20px;
  align-items:end;
}
.badge {
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.16);
  backdrop-filter: blur(10px);
  border-radius:999px;
  padding:10px 15px;
  font-weight:850;
  color:white;
}
.card {
  border: 1px solid rgba(255,255,255,.11);
  border-radius: 26px;
  background: linear-gradient(180deg, rgba(255,255,255,.070), rgba(255,255,255,.027));
  padding: 24px;
  box-shadow: 0 18px 55px rgba(0,0,0,.24);
}
.kicker {
  color:#9eb0bc;
  text-transform:uppercase;
  letter-spacing:.13em;
  font-weight:900;
  font-size:12px;
}
.big {
  font-size:38px;
  line-height:1.02;
  letter-spacing:-.050em;
  margin:10px 0 8px;
  font-weight:900;
  color:#f3f7f9;
}
.copy { color:#cbd8de; font-size:15.8px; line-height:1.58; }
.note {
  color:#9eb0bc;
  font-size:12.5px;
  line-height:1.45;
  border-top:1px solid rgba(255,255,255,.10);
  margin-top:14px;
  padding-top:12px;
}
.metric-row {
  display:flex;
  justify-content:space-between;
  gap:14px;
  border-top:1px solid rgba(255,255,255,.08);
  padding-top:12px;
  margin-top:12px;
}
.metric-label { color:#9eb0bc; font-size:13px; }
.metric-value { color:#f3f7f9; font-weight:850; }

.gauge-wrap {
  margin-top: 16px;
}
.gauge-track {
  width: 100%;
  height: 22px;
  border-radius: 999px;
  background: linear-gradient(90deg, #5ec9df, #f1d36b, #f2a34a, #e45f56, #7c4cc2);
  position: relative;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);
}
.gauge-marker {
  position:absolute;
  top:-9px;
  width: 4px;
  height: 40px;
  border-radius: 999px;
  background:#ffffff;
  box-shadow:0 0 0 5px rgba(255,255,255,.16), 0 8px 18px rgba(0,0,0,.35);
}
.gauge-labels {
  display:flex;
  justify-content:space-between;
  color:#9eb0bc;
  font-size:12px;
  margin-top:9px;
}
.pill {
  display:inline-block;
  padding:8px 12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.14);
  background:rgba(255,255,255,.07);
  color:#edf4f7;
  font-weight:800;
  margin:4px 6px 4px 0;
}
.meaning-grid {
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap:14px;
}
.meaning-card {
  border:1px solid rgba(255,255,255,.09);
  border-radius:18px;
  padding:16px;
  background:rgba(255,255,255,.035);
}
.meaning-card strong { color:#f3f7f9; }
.section-title {
  font-size:27px;
  letter-spacing:-.035em;
  margin: 0 0 10px;
  font-weight:900;
}
.small-muted { color:#9eb0bc; font-size:13px; }
@media (max-width: 900px) {
  .hero h1 { font-size:44px; }
  .hero-footer { position:static; margin-top:30px; flex-direction:column; align-items:flex-start; }
  .meaning-grid { grid-template-columns: 1fr; }
}


.landing-hero {
  min-height: 640px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 34px;
  padding: 54px;
  background:
    linear-gradient(90deg, rgba(3,10,15,.92), rgba(3,10,15,.62), rgba(3,10,15,.16)),
    var(--poudre-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 34px 100px rgba(0,0,0,.42);
  position: relative;
  overflow: hidden;
}
.landing-hero h1 {
  font-size: 54px;
  line-height: .92;
  letter-spacing: -.07em;
  margin: 20px 0 18px;
  max-width: 900px;
  color: #f6fbfd;
}
.landing-hero p {
  max-width: 650px;
  font-size: 18px;
  line-height: 1.38;
  color: #dfeaf0;
}
.landing-thesis {
  position: absolute;
  left: 54px;
  bottom: 48px;
  max-width: 720px;
  font-size: 22px;
  line-height: 1.18;
  letter-spacing: -.035em;
  font-weight: 850;
  color: #ffffff;
}
.landing-subline {
  color: #aebbc4;
  font-size: 14px;
  margin-top: 14px;
  letter-spacing: .02em;
  max-width: 700px;
}
.landing-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
  margin-top: 22px;
}
.landing-card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,.060), rgba(255,255,255,.025));
  padding: 22px;
  min-height: 160px;
  box-shadow: 0 18px 55px rgba(0,0,0,.22);
}
.landing-number {
  font-size: 12px;
  color: #5ec9df;
  font-weight: 900;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.landing-card h3 {
  font-size: 18px;
  line-height: 1.08;
  letter-spacing: -.035em;
  margin: 11px 0 9px;
  color: #f3f7f9;
}
.landing-card p {
  color: #cbd8de;
  line-height: 1.5;
  font-size: 15px;
}
.callout {
  margin-top: 22px;
  border: 1px solid rgba(94,201,223,.22);
  border-radius: 24px;
  padding: 26px;
  background: linear-gradient(135deg, rgba(94,201,223,.09), rgba(124,76,194,.06));
}
.callout h2 {
  font-size: 26px;
  letter-spacing: -.045em;
  margin: 0 0 10px;
}
.callout p {
  color: #d6e4ea;
  font-size: 17px;
  line-height: 1.5;
  max-width: 960px;
}

@media (max-width: 900px) {
  .landing-hero { min-height: 620px; padding: 32px; }
  .landing-hero h1 { font-size: 48px; }
  .landing-thesis { position: static; margin-top: 80px; font-size: 24px; }
  .landing-grid { grid-template-columns: 1fr; }
}


.public-hero {
  min-height: 640px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 34px;
  padding: 46px;
  background:
    linear-gradient(90deg, rgba(3,10,15,.94), rgba(3,10,15,.66), rgba(3,10,15,.20)),
    var(--poudre-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 34px 100px rgba(0,0,0,.42);
  position: relative;
  overflow: hidden;
}
.public-hero h1 {
  font-size: 48px;
  line-height: .98;
  letter-spacing: -.055em;
  margin: 18px 0 12px;
  max-width: 780px;
  color: #f6fbfd;
}
.public-hero p {
  max-width: 640px;
  font-size: 18px;
  line-height: 1.42;
  color: #dfeaf0;
}
.stress-panel {
  position:absolute;
  left:46px;
  bottom:42px;
  width:min(620px, calc(100% - 92px));
  border:1px solid rgba(255,255,255,.14);
  border-radius:26px;
  padding:24px;
  background:rgba(4,12,18,.66);
  backdrop-filter: blur(10px);
}
.stress-label {
  color:#9eb0bc;
  text-transform:uppercase;
  letter-spacing:.16em;
  font-weight:900;
  font-size:12px;
}
.stress-number {
  font-size:76px;
  line-height:.9;
  letter-spacing:-.06em;
  font-weight:950;
  margin:10px 0 10px;
  color:#f6fbfd;
}
.stress-track {
  height:18px;
  border-radius:999px;
  background:linear-gradient(90deg,#5ec9df,#f1d36b,#f2a34a,#e45f56,#7c4cc2);
  position:relative;
  margin:15px 0 8px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.18);
}
.stress-marker {
  position:absolute;
  top:-8px;
  width:4px;
  height:34px;
  border-radius:999px;
  background:white;
  box-shadow:0 0 0 5px rgba(255,255,255,.16), 0 8px 16px rgba(0,0,0,.35);
}
.stress-scale {
  display:flex;
  justify-content:space-between;
  color:#aebbc4;
  font-size:12px;
}
.public-grid {
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap:16px;
  margin-top:22px;
}
.public-card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,.060), rgba(255,255,255,.025));
  padding:20px;
  min-height:150px;
  box-shadow: 0 18px 55px rgba(0,0,0,.22);
}
.public-icon { font-size:28px; margin-bottom:10px; }
.public-card h3 {
  margin:0 0 8px;
  font-size:20px;
  letter-spacing:-.025em;
  color:#f3f7f9;
}
.public-card p {
  color:#cbd8de;
  line-height:1.45;
  font-size:14.5px;
  margin:0;
}
.public-explainer {
  margin-top:22px;
  border:1px solid rgba(94,201,223,.22);
  border-radius:24px;
  padding:26px;
  background: linear-gradient(135deg, rgba(94,201,223,.09), rgba(124,76,194,.06));
}
.public-explainer h2 {
  margin:0 0 10px;
  font-size:30px;
  letter-spacing:-.04em;
}
.public-explainer p {
  color:#d6e4ea;
  font-size:16px;
  line-height:1.55;
  max-width:980px;
}
@media (max-width: 900px) {
  .public-hero { min-height: 720px; padding: 28px; }
  .public-hero h1 { font-size: 38px; }
  .stress-panel { position:static; margin-top:70px; width:100%; }
  .stress-number { font-size:58px; }
  .public-grid { grid-template-columns: 1fr; }
}


.story-hero {
  min-height: 82vh;
  border-radius: 34px;
  padding: 46px;
  background:
    linear-gradient(90deg, rgba(3,10,15,.92), rgba(3,10,15,.60), rgba(3,10,15,.10)),
    var(--poudre-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 36px 110px rgba(0,0,0,.46);
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.12);
}
.story-eyebrow {
  color:#5ec9df;
  font-size:12px;
  letter-spacing:.18em;
  text-transform:uppercase;
  font-weight:900;
}
.story-title {
  font-size: clamp(44px, 6.2vw, 78px);
  line-height:.92;
  letter-spacing:-.07em;
  max-width: 900px;
  margin: 18px 0 16px;
  color:#f8fcff;
  font-weight:950;
}
.story-subtitle {
  max-width: 650px;
  font-size: 20px;
  line-height:1.42;
  color:#dfeaf0;
}
.story-stress-card {
  position:absolute;
  left:46px;
  bottom:42px;
  width:min(560px, calc(100% - 92px));
  padding:24px;
  border-radius:28px;
  background:rgba(5,14,21,.72);
  border:1px solid rgba(255,255,255,.16);
  backdrop-filter: blur(14px);
}
.story-stress-top {
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:18px;
}
.story-stress-number {
  font-size:78px;
  line-height:.86;
  letter-spacing:-.07em;
  font-weight:950;
  color:white;
}
.story-stress-label {
  font-size:12px;
  letter-spacing:.15em;
  text-transform:uppercase;
  color:#aebbc4;
  font-weight:900;
}
.story-stress-note {
  color:#d5e1e7;
  font-size:15px;
  line-height:1.45;
  margin-top:12px;
}
.story-track {
  height:16px;
  border-radius:999px;
  background:linear-gradient(90deg,#5ec9df,#f1d36b,#f2a34a,#e45f56,#7c4cc2);
  margin:16px 0 8px;
  position:relative;
}
.story-marker {
  position:absolute;
  top:-8px;
  width:4px;
  height:32px;
  border-radius:999px;
  background:white;
  box-shadow:0 0 0 5px rgba(255,255,255,.17), 0 10px 22px rgba(0,0,0,.4);
}
.story-scale {
  display:flex;
  justify-content:space-between;
  color:#aebbc4;
  font-size:12px;
}
.story-section {
  margin-top:28px;
  padding:38px;
  border-radius:30px;
  border:1px solid rgba(255,255,255,.10);
  background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.024));
  box-shadow:0 24px 70px rgba(0,0,0,.24);
}
.story-section h2 {
  margin:0 0 12px;
  color:#f6fbfd;
  font-size:38px;
  line-height:1;
  letter-spacing:-.055em;
}
.story-section p {
  color:#d5e1e7;
  font-size:18px;
  line-height:1.55;
  max-width:950px;
}
.tile-grid {
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:18px;
  margin-top:22px;
}
.visual-tile {
  min-height:240px;
  border-radius:26px;
  overflow:hidden;
  position:relative;
  border:1px solid rgba(255,255,255,.11);
  background-size:cover;
  background-position:center;
  box-shadow:0 18px 60px rgba(0,0,0,.28);
}
.visual-tile::after {
  content:"";
  position:absolute;
  inset:0;
  background:linear-gradient(180deg, rgba(0,0,0,.10), rgba(0,0,0,.72));
}
.tile-content {
  position:absolute;
  left:22px;
  right:22px;
  bottom:20px;
  z-index:2;
}
.tile-content h3 {
  margin:0 0 8px;
  color:white;
  font-size:26px;
  line-height:1.05;
  letter-spacing:-.04em;
}
.tile-content p {
  margin:0;
  color:#d9e6eb;
  font-size:15px;
  line-height:1.42;
}
.big-idea {
  margin-top:28px;
  border-radius:32px;
  padding:42px;
  background:
    radial-gradient(circle at 12% 12%, rgba(94,201,223,.16), transparent 32%),
    linear-gradient(135deg, rgba(18,31,42,.86), rgba(15,20,29,.86));
  border:1px solid rgba(94,201,223,.18);
}
.big-idea h2 {
  margin:0;
  color:#f8fcff;
  font-size:46px;
  line-height:1.02;
  letter-spacing:-.06em;
  max-width:900px;
}
.big-idea p {
  color:#d5e1e7;
  font-size:18px;
  line-height:1.55;
  max-width:900px;
}
.story-button {
  display:inline-block;
  margin-top:18px;
  padding:12px 18px;
  border-radius:999px;
  color:#071018;
  background:#5ec9df;
  font-weight:900;
}
@media (max-width:900px) {
  .story-hero { min-height:760px; padding:28px; }
  .story-stress-card { position:static; margin-top:90px; width:100%; }
  .tile-grid { grid-template-columns:1fr; }
  .story-section, .big-idea { padding:26px; }
}


.rights-hero {
  min-height: 84vh;
  border-radius: 34px;
  padding: 48px;
  background:
    linear-gradient(90deg, rgba(3,10,15,.94), rgba(3,10,15,.64), rgba(3,10,15,.12)),
    var(--poudre-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 36px 110px rgba(0,0,0,.46);
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.12);
}
.rights-title {
  font-size: clamp(46px, 6.5vw, 84px);
  line-height: .90;
  letter-spacing: -.075em;
  max-width: 980px;
  margin: 18px 0 18px;
  color:#f8fcff;
  font-weight:950;
}
.rights-subtitle {
  max-width: 720px;
  font-size: 23px;
  line-height:1.36;
  color:#dfeaf0;
}
.rights-bottom {
  position:absolute;
  left:48px;
  right:48px;
  bottom:42px;
  display:grid;
  grid-template-columns: 1fr 360px;
  gap:28px;
  align-items:end;
}
.rights-thesis {
  max-width:820px;
  color:white;
  font-size:30px;
  line-height:1.16;
  letter-spacing:-.04em;
  font-weight:850;
}
.rights-mini-panel {
  border:1px solid rgba(255,255,255,.16);
  border-radius:26px;
  padding:22px;
  background:rgba(4,12,18,.68);
  backdrop-filter: blur(12px);
}
.rights-mini-label {
  color:#9eb0bc;
  text-transform:uppercase;
  letter-spacing:.16em;
  font-weight:900;
  font-size:12px;
}
.rights-mini-number {
  color:white;
  font-size:56px;
  line-height:.9;
  letter-spacing:-.06em;
  font-weight:950;
  margin:10px 0;
}
.rights-section {
  margin-top:28px;
  padding:38px;
  border-radius:30px;
  border:1px solid rgba(255,255,255,.10);
  background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.024));
  box-shadow:0 24px 70px rgba(0,0,0,.24);
}
.rights-section h2 {
  margin:0 0 12px;
  color:#f6fbfd;
  font-size:40px;
  line-height:1;
  letter-spacing:-.055em;
}
.rights-section p {
  color:#d5e1e7;
  font-size:18px;
  line-height:1.55;
  max-width:980px;
}
.rights-grid {
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:18px;
  margin-top:22px;
}
.rights-card {
  border:1px solid rgba(255,255,255,.10);
  border-radius:26px;
  padding:24px;
  min-height:210px;
  background:linear-gradient(180deg, rgba(255,255,255,.060), rgba(255,255,255,.025));
  box-shadow:0 18px 55px rgba(0,0,0,.22);
}
.rights-icon { font-size:34px; margin-bottom:14px; }
.rights-card h3 {
  margin:0 0 10px;
  color:white;
  font-size:27px;
  letter-spacing:-.04em;
  line-height:1.05;
}
.rights-card p {
  margin:0;
  color:#cbd8de;
  font-size:15.5px;
  line-height:1.5;
}
.call-timeline {
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:12px;
  margin-top:26px;
}
.call-step {
  border-radius:22px;
  padding:20px;
  min-height:130px;
  border:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.04);
}
.call-step strong {
  display:block;
  font-size:20px;
  color:white;
  margin-bottom:8px;
}
.call-step span {
  color:#cbd8de;
  font-size:14.5px;
  line-height:1.4;
}
.rights-cta {
  margin-top:28px;
  border-radius:32px;
  padding:42px;
  background:
    radial-gradient(circle at 12% 12%, rgba(94,201,223,.16), transparent 32%),
    linear-gradient(135deg, rgba(18,31,42,.86), rgba(15,20,29,.86));
  border:1px solid rgba(94,201,223,.18);
}
.rights-cta h2 {
  margin:0;
  color:#f8fcff;
  font-size:46px;
  line-height:1.02;
  letter-spacing:-.06em;
  max-width:900px;
}
.rights-cta p {
  color:#d5e1e7;
  font-size:18px;
  line-height:1.55;
  max-width:900px;
}
@media (max-width:900px) {
  .rights-hero { min-height:800px; padding:30px; }
  .rights-bottom { position:static; margin-top:80px; grid-template-columns:1fr; }
  .rights-thesis { font-size:24px; }
  .rights-grid, .call-timeline { grid-template-columns:1fr; }
  .rights-section, .rights-cta { padding:26px; }
}


.beta-hero{min-height:84vh;border-radius:36px;padding:48px;background:linear-gradient(90deg,rgba(3,10,15,.95),rgba(3,10,15,.66),rgba(3,10,15,.18)),var(--poudre-bg);background-size:cover;background-position:center;border:1px solid rgba(255,255,255,.12);box-shadow:0 38px 120px rgba(0,0,0,.48);position:relative;overflow:hidden}
.beta-brand{color:#5ec9df;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:950}
.beta-title{max-width:980px;margin:18px 0 12px;color:#f8fcff;font-weight:950;font-size:clamp(44px,5.8vw,76px);line-height:.92;letter-spacing:-.075em}
.beta-subtitle{max-width:720px;color:#dfeaf0;font-size:22px;line-height:1.36}
.beta-status{position:absolute;left:48px;right:48px;bottom:42px;display:grid;grid-template-columns:1.05fr .95fr;gap:22px;align-items:stretch}
.beta-panel{border:1px solid rgba(255,255,255,.15);border-radius:28px;padding:24px;background:rgba(4,12,18,.72);backdrop-filter:blur(14px)}
.beta-panel-label{color:#aebbc4;font-size:12px;letter-spacing:.15em;text-transform:uppercase;font-weight:900}
.beta-score{font-size:76px;line-height:.88;letter-spacing:-.07em;font-weight:950;color:white;margin:10px 0 12px}
.beta-track{height:16px;border-radius:999px;background:linear-gradient(90deg,#5ec9df,#f1d36b,#f2a34a,#e45f56,#7c4cc2);position:relative;margin:14px 0 8px}
.beta-marker{position:absolute;top:-8px;height:32px;width:4px;border-radius:999px;background:white;box-shadow:0 0 0 5px rgba(255,255,255,.17),0 10px 22px rgba(0,0,0,.42)}
.beta-scale{display:flex;justify-content:space-between;color:#aebbc4;font-size:12px}
.beta-narrative{color:#e3edf2;font-size:20px;line-height:1.42;letter-spacing:-.02em;font-weight:650}
.beta-meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.beta-pill{display:inline-block;padding:8px 11px;border-radius:999px;border:1px solid rgba(255,255,255,.13);background:rgba(255,255,255,.08);color:#e7f0f4;font-size:13px;font-weight:800}
.beta-section{margin-top:28px;padding:36px;border-radius:30px;border:1px solid rgba(255,255,255,.10);background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.024));box-shadow:0 24px 70px rgba(0,0,0,.24)}
.beta-section h2{margin:0 0 12px;color:#f6fbfd;font-size:38px;line-height:1;letter-spacing:-.055em}
.beta-section p{color:#d5e1e7;font-size:17px;line-height:1.55;max-width:980px}
.beta-map-wrap{display:grid;grid-template-columns:.9fr 1.1fr;gap:24px;align-items:center}
.beta-map{width:100%;max-width:540px;margin:auto;display:block}
.beta-flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:24px}
.beta-flow-step{border-radius:22px;padding:18px;border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.045);min-height:136px}
.beta-flow-step strong{color:white;display:block;font-size:18px;line-height:1.08;letter-spacing:-.025em;margin-bottom:8px}
.beta-flow-step span{color:#cbd8de;font-size:14px;line-height:1.38}
.beta-grid-four{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:22px}
.beta-impact{border:1px solid rgba(255,255,255,.10);border-radius:24px;padding:20px;background:rgba(255,255,255,.04);min-height:160px}
.beta-impact .icon{font-size:30px;margin-bottom:10px}.beta-impact h3{margin:0 0 8px;color:white;font-size:22px;letter-spacing:-.035em}.beta-impact p{margin:0;color:#cbd8de;font-size:14.5px;line-height:1.45}
@media(max-width:900px){.beta-hero{min-height:920px;padding:30px}.beta-status{position:static;margin-top:70px;grid-template-columns:1fr}.beta-score{font-size:58px}.beta-map-wrap,.beta-flow,.beta-grid-four{grid-template-columns:1fr}.beta-section{padding:26px}}

</style>
""", unsafe_allow_html=True)

def apply_expert_probability_overlay(prob_series, current_regime, selected_date):
    adjusted = prob_series.copy().astype(float)
    month = int(pd.Timestamp(selected_date).month)
    irrigation_admin_season = 4 <= month <= 10
    active_admin = current_regime != "Free River"
    if irrigation_admin_season and active_admin:
        adjusted.loc["Free River"] = 0.0
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
    return adjusted

@st.cache_data
def load_data():
    daily = pd.read_csv("data/SouthPlatte_Basin_Daily_Regime_Index.csv", parse_dates=["date"])
    annual = pd.read_csv("data/SouthPlatte_Basin_Annual_Regime_Days.csv")
    flow = pd.read_csv("data/Selected_Station_Analysis_Xtab_202605281804.csv")
    flow["date"] = pd.to_datetime(flow["meas_date"], errors="coerce")
    flow = flow.rename(columns={"Streamflow Value": "flow_cfs"})[["date", "flow_cfs"]]
    flow["flow_cfs"] = pd.to_numeric(flow["flow_cfs"], errors="coerce")
    return daily, annual, flow

def prepare_model_dataset(daily, flow):
    df = daily.merge(flow, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["flow_cfs"] = pd.to_numeric(df["flow_cfs"], errors="coerce").interpolate(limit_direction="both")
    df["flow_7d"] = df["flow_cfs"].rolling(7, min_periods=1).mean()
    df["flow_14d"] = df["flow_cfs"].rolling(14, min_periods=1).mean()
    df["flow_30d"] = df["flow_cfs"].rolling(30, min_periods=1).mean()
    df["flow_change_7d"] = df["flow_cfs"] - df["flow_cfs"].shift(7)
    df["flow_change_14d"] = df["flow_cfs"] - df["flow_cfs"].shift(14)

    df["flow_doy_percentile"] = np.nan
    for doy in range(1, 367):
        window = list(range(max(1, doy-7), min(366, doy+7)+1))
        hist = df[df["doy"].isin(window)]["flow_cfs"].dropna()
        mask = df["doy"] == doy
        if len(hist) > 10:
            df.loc[mask, "flow_doy_percentile"] = df.loc[mask, "flow_cfs"].apply(lambda x: (hist <= x).mean() * 100)

    df["severity_change_7d"] = df["severity"] - df["severity"].shift(7)
    df["severity_change_14d"] = df["severity"] - df["severity"].shift(14)
    regime_change = (df["regime"] != df["regime"].shift(1)).cumsum()
    df["days_in_current_regime"] = df.groupby(regime_change).cumcount() + 1
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 366)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 366)
    df["target_regime_30d"] = df["regime"].shift(-30)
    return df

FEATURES = [
    "severity", "score", "historical_percentile", "month", "doy", "sin_doy", "cos_doy",
    "flow_cfs", "flow_7d", "flow_14d", "flow_30d", "flow_change_7d", "flow_change_14d",
    "flow_doy_percentile", "severity_change_7d", "severity_change_14d", "days_in_current_regime",
]

@st.cache_resource
def train_model(model_df):
    train_df = model_df.dropna(subset=FEATURES + ["target_regime_30d"]).copy()
    le = LabelEncoder()
    le.fit(REGIME_ORDER)
    y = le.transform(train_df["target_regime_30d"])
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(train_df[FEATURES], y)
    return clf, le

def comparable_years(model_df, selected_date, score):
    doy = int(selected_date.dayofyear)
    y0 = int(selected_date.year)
    comps = []
    for y, g in model_df.groupby(model_df["date"].dt.year):
        if y == y0:
            continue
        w = g[g["doy"].between(max(1, doy-30), doy)]
        if len(w) > 10:
            comps.append((y, abs(w["score"].mean() - score), w["score"].mean()))
    comps = sorted(comps, key=lambda x: x[1])[:3]
    return [str(x[0]) for x in comps]


@st.cache_data(show_spinner=False)
def build_stochastic_hydrology_model(model_df):
    df = model_df.sort_values("date").copy()
    if "flow_cfs" not in df.columns:
        df["flow_cfs"] = 0.0
    if "doy" not in df.columns:
        df["doy"] = df["date"].dt.dayofyear
    if "score" not in df.columns:
        df["score"] = 0.0

    df["flow_cfs"] = pd.to_numeric(df["flow_cfs"], errors="coerce").interpolate(limit_direction="both").fillna(0)
    df["log_flow"] = np.log1p(df["flow_cfs"].clip(lower=0))

    autocorr = []
    for lag in [1, 3, 7, 14, 30]:
        autocorr.append({
            "Variable": "Poudre canyon-mouth flow",
            "Lag Days": lag,
            "Autocorrelation": df["log_flow"].autocorr(lag=lag)
        })

    crosscorr = []
    for lag in [0, 3, 7, 14, 30, 45, 60]:
        crosscorr.append({
            "Signal": "Flow leading call-depth score",
            "Lead/Lag Days": lag,
            "Correlation": df["log_flow"].shift(lag).corr(pd.to_numeric(df["score"], errors="coerce"))
        })

    seasonal = df.groupby("doy")["log_flow"].mean().to_dict()
    df["seasonal_log_flow"] = df["doy"].map(seasonal)
    df["flow_anom"] = df["log_flow"] - df["seasonal_log_flow"]

    valid = pd.DataFrame({"lag": df["flow_anom"].shift(1), "now": df["flow_anom"]}).dropna()
    if len(valid) > 10 and valid["lag"].var() > 0:
        phi = float(np.cov(valid["lag"], valid["now"])[0, 1] / np.var(valid["lag"]))
    else:
        phi = 0.85
    phi = max(min(phi, 0.98), -0.2)

    resid = valid["now"] - phi * valid["lag"] if len(valid) else pd.Series(dtype=float)
    sigma = float(resid.std()) if len(resid) else 0.15
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.15

    return {
        "autocorr": pd.DataFrame(autocorr),
        "crosscorr": pd.DataFrame(crosscorr),
        "phi": phi,
        "sigma": sigma,
        "seasonal_log_flow": seasonal,
    }


def simulate_stochastic_outlook(model_df, clf, le, base_row, n_sims=150, horizons=(30, 60, 90), seed=42):
    rng = np.random.default_rng(seed)
    hydro = build_stochastic_hydrology_model(model_df)
    phi = hydro["phi"]
    sigma = hydro["sigma"]
    seasonal = hydro["seasonal_log_flow"]

    base = base_row.iloc[0].copy()
    base_date = pd.Timestamp(base.get("date", pd.Timestamp.today())).normalize()
    base_flow = float(base.get("flow_cfs", model_df["flow_cfs"].median() if "flow_cfs" in model_df.columns else 0) or 0)
    base_log_flow = np.log1p(max(base_flow, 0))
    base_doy = int(base.get("doy", base_date.dayofyear))
    base_seasonal = seasonal.get(base_doy, base_log_flow)
    anom = np.full(n_sims, base_log_flow - base_seasonal, dtype=float)

    max_h = max(horizons)
    sim_flows = np.zeros((n_sims, max_h), dtype=float)

    for day in range(1, max_h + 1):
        future_date = base_date + pd.Timedelta(days=day)
        future_doy = int(future_date.dayofyear)
        season_mean = seasonal.get(future_doy, base_seasonal)
        anom = phi * anom + rng.normal(0, sigma, size=n_sims)
        sim_flows[:, day - 1] = np.maximum(np.expm1(season_mean + anom), 0)

    feature_medians = model_df[FEATURES].median(numeric_only=True)
    outputs = []

    for h in horizons:
        future_date = base_date + pd.Timedelta(days=h)
        future_doy = int(future_date.dayofyear)
        rows = []

        for i in range(n_sims):
            row = base.copy()
            sim_flow = float(sim_flows[i, h - 1])
            recent = sim_flows[i, :h]

            row["date"] = future_date
            row["year"] = future_date.year
            row["month"] = future_date.month
            row["doy"] = future_doy
            row["flow_cfs"] = sim_flow
            row["flow_7d"] = float(np.mean(recent[-7:]))
            row["flow_14d"] = float(np.mean(recent[-14:]))
            row["flow_30d"] = float(np.mean(recent[-30:]))
            row["flow_change_7d"] = sim_flow - base_flow
            row["flow_change_14d"] = sim_flow - base_flow
            row["sin_doy"] = np.sin(2 * np.pi * future_doy / 366)
            row["cos_doy"] = np.cos(2 * np.pi * future_doy / 366)
            rows.append(row)

        Xh = pd.DataFrame(rows)[FEATURES].fillna(feature_medians)
        preds = clf.predict(Xh)
        regimes = le.inverse_transform(preds)

        for regime, sim_flow in zip(regimes, sim_flows[:, h - 1]):
            outputs.append({
                "horizon_days": h,
                "predicted_regime": regime,
                "simulated_flow_cfs": float(sim_flow),
            })

    out = pd.DataFrame(outputs)
    probs = (
        out.groupby(["horizon_days", "predicted_regime"])
        .size()
        .div(out.groupby("horizon_days").size(), level=0)
        .reset_index(name="probability")
    )

    return probs, hydro


daily, annual, flow = load_data()
model_df = prepare_model_dataset(daily, flow)
clf, le = train_model(model_df)

page = "Outlook Dashboard"
latest_available = daily["date"].max()
today_date = pd.Timestamp.today().normalize()
default_date = today_date



def get_model_row_for_date(model_df, selected_date):
    selected_date = pd.Timestamp(selected_date).normalize()
    exact = model_df[model_df["date"] == selected_date]
    if not exact.empty:
        return exact.iloc[-1:]
    same_doy = model_df[model_df["doy"] == selected_date.dayofyear]
    if not same_doy.empty:
        return same_doy.sort_values("date").iloc[[-1]]
    return model_df.sort_values("date").iloc[[-1]]

st.sidebar.title("Scenario")

st.markdown("""
<style>
/* Final cleanup: remove decorative long rounded dividers/capsules */
.metric-row,
.pill,
.beta-pill,
.beta-meta,
.status-pill,
.badge,
.capsule,
.stat-bar,
.metric-bar,
.metric-top,
.metric-header,
.hero-footer {
    display: none !important;
}

/* Prevent empty markdown blocks/divs from becoming visible rounded bars */
div:empty {
    display: none !important;
}

/* Keep metric columns flat */
[data-testid="column"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* But keep Plotly charts visible */
.js-plotly-plot,
.plotly,
[data-testid="stPlotlyChart"] {
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Flatten the top snapshot area: no rounded horizontal bubble cards above data */
[data-testid="column"] > div:has(> div > div > p) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
hr {
    opacity: 0.55;
}
</style>
""", unsafe_allow_html=True)

use_live_dashboard = True
date_choice = st.sidebar.date_input(
    "Choose a historical date",
    value=default_date.date(),
    min_value=daily["date"].min().date(),
    max_value=max(latest_available, today_date).date(),
)
selected_date = pd.Timestamp(date_choice)




row = get_model_row_for_date(model_df, selected_date)

X = row[FEATURES].fillna(model_df[FEATURES].median(numeric_only=True))
prob = clf.predict_proba(X)[0]
raw_prob_series = pd.Series(prob, index=le.inverse_transform(np.arange(len(prob)))).reindex(REGIME_ORDER, fill_value=0)

current_regime = row.iloc[0]["regime"]
prob_series = apply_expert_probability_overlay(raw_prob_series, current_regime, selected_date).sort_values(ascending=False)

if use_live_dashboard:
    try:
        live_calls_dash, live_url_dash, live_msg_dash = fetch_live_active_calls()
        live_state_dash = build_live_current_state(live_calls_dash)
        row, raw_prob_series, prob_series = build_forecast_from_state(model_df, clf, le, live_state_dash, selected_date)
        current_regime = live_state_dash["regime"]
        row.loc[row.index[0], "controlling_priority_date"] = live_state_dash["controlling_priority_date"]
        row.loc[row.index[0], "controlling_priority_structure"] = live_state_dash["controlling_priority_structure"]
    except Exception as e:
        st.sidebar.warning(f"Live CDSS unavailable; using historical selected date. {e}")

hist_pct = float(row.iloc[0]["historical_percentile"])
score = float(row.iloc[0]["score"])
priority = format_call_date_mmddyyyy(row.iloc[0].get("controlling_priority_date", "—"))
structure = row.iloc[0].get("controlling_priority_structure", "—")
flow_cfs = float(row.iloc[0].get("flow_cfs", np.nan))
flow_pct = float(row.iloc[0].get("flow_doy_percentile", np.nan))
comps = comparable_years(model_df, selected_date, score)

public_label = PUBLIC_LABELS[current_regime]
public_explain = PUBLIC_EXPLAIN[current_regime]
most_likely = prob_series.index[0]
most_likely_public = PUBLIC_LABELS[most_likely]

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Historical-data prototype</div>
  <h1>Water Right Outlook</h1>
  <p>Making Colorado water rights understandable — by translating river administration into a plain-English outlook.</p>
</div>
</div>
""", unsafe_allow_html=True)



st.markdown("### Current Water Right Snapshot")

top1, top2, top3 = st.columns(3)

with top1:
    st.markdown('<div class="kicker">Current Condition</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="big" style="color:{REGIME_COLORS[current_regime]};">{public_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="copy">{public_explain}</div>', unsafe_allow_html=True)

with top2:
    st.markdown('<div class="kicker">Current Call</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="big" style="font-size:32px;">{priority}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="copy">{structure}</div>', unsafe_allow_html=True)

with top3:
    st.markdown('<div class="kicker">30-Day Outlook</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="big" style="color:{REGIME_COLORS[most_likely]};">{most_likely_public}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="copy">Most likely outcome: {prob_series.iloc[0]:.0%} probability.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.08, .92], gap="large")
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What could the basin look like in 30 days?</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for regime in REGIME_ORDER:
        public = PUBLIC_SHORT[regime]
        fig.add_trace(go.Bar(
            x=[prob_series.get(regime, 0)],
            y=[public],
            orientation="h",
            marker_color=REGIME_COLORS[regime],
            text=[f"{prob_series.get(regime, 0):.0%}"],
            textposition="outside",
            hovertemplate=f"{PUBLIC_LABELS[regime]}<br>%{{x:.0%}}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=35, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(categoryorder="array", categoryarray=[PUBLIC_SHORT[r] for r in reversed(REGIME_ORDER)]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4f7", size=14),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">The displayed forecast includes an expert-rule overlay: if active administration is underway during irrigation season, Free River is removed as a 30-day outcome.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="meaning-grid" style="grid-template-columns:1fr;">
      <div class="meaning-card"><strong>For the public</strong><br><span class="copy">This is like a water-right outlook: not just how much water is in the river, but how the river is being administered.</span></div>
      <div class="meaning-card"><strong>For water-right owners</strong><br><span class="copy">The outlook gives a plain-English signal of whether administrative pressure is likely to ease, hold, or tighten.</span></div>
      <div class="meaning-card"><strong>For Northern Colorado</strong><br><span class="copy">The key issue is not only drought. It is whether priority administration becomes more restrictive.</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")


st.markdown("### Supporting Hydrology and Context")

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown('<div class="kicker">Canyon-Mouth Flow</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{flow_cfs:,.0f} cfs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Seasonal flow percentile: {flow_pct:.0f}%</div>', unsafe_allow_html=True)

with h2:
    st.markdown('<div class="kicker">Comparable Years</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{", ".join(comps) if comps else "—"}</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">Based on recent administrative severity near this point in the season.</div>', unsafe_allow_html=True)

with h3:
    st.markdown('<div class="kicker">Severity Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{score:.0f}/100</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">Higher means more restrictive water-right administration.</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### 30-Day Water Rights Outlook")

outlook_df = (
    prob_series
    .reindex(REGIME_ORDER)
    .reset_index()
    .rename(columns={"index": "Regime", 0: "Probability"})
)
outlook_df["Public Label"] = outlook_df["Regime"].map(PUBLIC_LABELS)
outlook_df["Short Label"] = outlook_df["Regime"].map(PUBLIC_SHORT)
outlook_df["Percent"] = outlook_df["Probability"] * 100

top_regime = prob_series.idxmax()
top_prob = prob_series.max()

c_out1, c_out2 = st.columns([0.72, 0.28], gap="large")

with c_out1:
    fig = go.Figure()
    for regime in REGIME_ORDER:
        row_prob = float(outlook_df.loc[outlook_df["Regime"] == regime, "Probability"].iloc[0])
        fig.add_trace(go.Bar(
            x=[row_prob],
            y=[PUBLIC_SHORT[regime]],
            orientation="h",
            marker_color=REGIME_COLORS[regime],
            text=[f"{row_prob:.0%}"],
            textposition="outside",
            hovertemplate=f"{PUBLIC_LABELS[regime]}<br>%{{x:.0%}}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            title="Probability",
            gridcolor="rgba(255,255,255,.10)"
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=[PUBLIC_SHORT[r] for r in reversed(REGIME_ORDER)]
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4f7", size=14),
    )
    st.plotly_chart(fig, use_container_width=True, key="thirty_day_probability_outlook")

with c_out2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Most likely outcome</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="big" style="color:{REGIME_COLORS[top_regime]};">{PUBLIC_LABELS[top_regime]}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="copy">{top_prob:.0%} probability over the next 30 days.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="note">This forecast combines the historical regime model with expert rules for realistic water-right outcomes.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)




st.markdown("### Annual regime history: 2005–present")
annual = annual.rename(columns={annual.columns[0]: "year"}) if annual.columns[0] != "year" else annual
plot_annual = annual[annual["year"] >= 2005].copy()

fig_annual = go.Figure()
for regime in REGIME_ORDER:
    if regime in plot_annual.columns:
        fig_annual.add_trace(go.Bar(
            x=plot_annual["year"].astype(str),
            y=plot_annual[regime],
            name=PUBLIC_SHORT[regime],
            marker_color=REGIME_COLORS[regime],
            hovertemplate=f"{PUBLIC_LABELS[regime]}<br>%{{y}} days<extra></extra>",
        ))

fig_annual.update_layout(
    barmode="stack",
    height=450,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Days", gridcolor="rgba(255,255,255,.10)"),
    xaxis=dict(title="Year"),
    legend=dict(orientation="h", y=1.13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf4f7"),
)
st.plotly_chart(fig_annual, use_container_width=True, key="annual_regime_history_2005_present")


with st.expander("About this prototype"):
    st.write("""
This historical-data prototype avoids live CDSS API calls so it deploys reliably.
The date selector simulates what the site would have shown on a selected historical date.

The public labels are simplified translations of the technical administrative regimes:
Free River, Mild Administration, Normal Administration, Senior Administration, and Exceptional Administration.
""")
    compare = pd.DataFrame({
        "Raw model probability": raw_prob_series.reindex(REGIME_ORDER),
        "Expert-adjusted probability": prob_series.reindex(REGIME_ORDER),
    })
    compare.index = [PUBLIC_LABELS[i] for i in compare.index]
    st.dataframe(compare.style.format("{:.1%}"), use_container_width=True)
    st.dataframe(row, use_container_width=True)

st.markdown('<div class="note">Not an official forecast, legal opinion, or substitute for CDSS/DWR records. Built as a public-education prototype.</div>', unsafe_allow_html=True)
def get_model_row_for_date(model_df, selected_date):
    """
    Return an exact model row if available. If the user selects today's date
    and the historical snapshot is not updated through today, use the closest
    available date with the same day-of-year; otherwise fall back to latest row.
    """
    selected_date = pd.Timestamp(selected_date)
    exact = model_df[model_df["date"] == selected_date]
    if not exact.empty:
        return exact.iloc[-1:]

    same_doy = model_df[model_df["doy"] == selected_date.dayofyear]
    if not same_doy.empty:
        # Prefer the most recent historical year with the same day of year.
        return same_doy.sort_values("date").iloc[[-1]]

    return model_df.iloc[[-1]]


