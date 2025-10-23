from __future__ import annotations

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mplsoccer import Radar


from datetime import datetime, date

def _player_age(row):
    # Try common age fields
    for key in ["Age", "age"]:
        if key in row and row[key] is not None:
            try:
                val = int(float(row[key]))
                if val > 0:
                    return val
            except Exception:
                pass
    # Try DOB parsing (YYYY-MM-DD or DD/MM/YYYY etc.)
    for key in ["DOB", "Date of Birth", "Birthdate", "Birth Date"]:
        if key in row and isinstance(row[key], str) and row[key].strip():
            txt = row[key].strip()
            # try multiple formats
            fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]
            for fmt in fmts:
                try:
                    dob = datetime.strptime(txt, fmt).date()
                    today = date.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    if age > 0:
                        return age
                except Exception:
                    continue
    # Try Birth Year
    for key in ["Birth Year", "Birth year", "YOB", "Year of Birth"]:
        if key in row and row[key] is not None:
            try:
                year = int(float(row[key]))
                today = date.today()
                age = today.year - year
                if 10 < age < 60:
                    return age
            except Exception:
                pass
    return None



def _fix_npxg_block(df):
    import numpy as np
    if "xG" in df.columns and "Penalties taken" in df.columns:
        df["npxG"] = df["xG"].fillna(0) - df["Penalties taken"].fillna(0) * 0.81
    # per 90
    if "npxG" in df.columns and "Minutes played" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            mp = df["Minutes played"].replace(0, np.nan).astype(float)
            df["npxG per 90"] = (df["npxG"].astype(float) / mp) * 90.0
    # per shot
    if "npxG" in df.columns and "Shots" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["npxG per Shot"] = df["npxG"].astype(float) / df["Shots"].replace(0, np.nan).astype(float)
    # G-xG
    if "Goals" in df.columns and "xG" in df.columns:
        df["G-xG"] = df["Goals"].fillna(0) - df["xG"].fillna(0)
    return df


# ===================== CONFIG & STYLE =====================
st.set_page_config(
    page_title="Composite Metrics & Radar",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .metric-card {padding: 1rem; border: 1px solid rgba(49,51,63,0.2); border-radius: 12px;}
    .section {margin-top: .75rem; margin-bottom: .25rem; font-weight: 600; opacity: .9}
    .subtle {color: rgba(49,51,63,0.6)}
    .stTabs [data-baseweb="tab-list"] {gap: 12px}
    .stTabs [data-baseweb="tab"] {background: #f6f6f9; padding: 8px 14px; border-radius: 10px;}
    .small {font-size: 0.9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== CONSTANTS =====================
OFFENSIVE_COMPONENTS = [
    "Successful attacking actions per 90",
    "xG per 90",
    "xA per 90",
    "Key passes per 90",
    "Deep completions per 90",
    "Deep completed crosses per 90",
    "Progressive runs per 90",
    "Passes to penalty area per 90",
    "Smart passes per 90",
    "Crosses to goalie box per 90",
    "Touches in box per 90",
]
DEFAULT_OFF_WEIGHTS = {
    "Successful attacking actions per 90": 1.0,
    "xG per 90": 1.0,
    "xA per 90": 1.0,
    "Key passes per 90": 1.0,
    "Deep completions per 90": 1.0,
    "Deep completed crosses per 90": 0.8,
    "Progressive runs per 90": 0.6,
    "Passes to penalty area per 90": 0.8,
    "Smart passes per 90": 0.5,
    "Crosses to goalie box per 90": 0.4,
    "Touches in box per 90": 0.4,
}
PHYS_COLS = {
    "distance_total_m": "Distance P90",
    "running_m": "Running Distance P90",
    "hi_m": "HI Distance P90",
    "hsr_m": "HSR Distance P90",
    "sprint_m": "Sprint Distance P90",
    "hsr_cnt": "HSR Count P90",
    "sprint_cnt": "Sprint Count P90",
    "expl_hsr_cnt": "Explosive Acceleration to HSR Count P90",
    "expl_sprint_cnt": "Explosive Acceleration to Sprint Count P90",
}
DEF_COLS = [
    "Successful defensive actions per 90",
    "Defensive duels per 90",
    "Defensive duels won, %",
    "Aerial duels per 90",
    "Aerial duels won, %",
    "PAdj Sliding tackles",
    "PAdj Interceptions",
    "Shots blocked per 90",
]
GK_METRICS = [
    "Save rate, %","Prevented goals per 90","Conceded goals per 90","Shots against per 90","Clean sheets",
    "Back passes received as GK per 90","xG against","xG against per 90","Prevented goals","Shots against",
    "Conceded goals","Exits per 90",
]
ID_COLS_CANDIDATES = ["Player", "Short Name", "Team", "Position", "Minutes played", "Minutes"]

PRESETS = {
    "forward": [
        "npxG per 90","xG per 90","Shots per 90","Shots on target, %",
        "xA per 90","Key passes per 90","Touches in box per 90",
        "Progressive runs per 90","Deep completions per 90",
        "Finishing","Poaching","Aerial Threat",
        "Work Rate Offensive","Offensive Intensity","Offensive Explosion",
    ],
    "winger": [
        "Dribbles per 90","Dribbles won, %","Crosses per 90","Accurate crosses, %",
        "Deep completed crosses per 90","Progressive runs per 90","xA per 90","Key passes per 90",
        "Creativity","Progression",
        "Work Rate Offensive","Offensive Intensity","Offensive Explosion",
    ],
    "attacking_midfielder": [
        "Creativity","Progression","xA per 90","Key passes per 90","Smart passes per 90",
        "Deep completions per 90","xG Buildup",
        "Work Rate Offensive","Offensive Intensity","Offensive Explosion",
    ],
    "central_midfielder": [
        "Progression","Passing Quality","Creativity","xG Buildup",
        "PAdj Interceptions","Successful defensive actions per 90",
        "Work Rate Defensive","Defensive Intensity",
    ],
    "defensive_midfielder": [
        "Defence","Progression","Passing Quality","Discipline","Involvement",
        "xG Buildup","PAdj Interceptions","Successful defensive actions per 90",
        "Work Rate Defensive","Defensive Intensity","Defensive Explosion",
    ],
    "full_back": [
        "Progression","Creativity","Passing Quality","Defence","Aerial Defence",
        "Deep completed crosses per 90","Progressive runs per 90","Crosses per 90",
        "Work Rate Defensive","Defensive Intensity","Defensive Explosion",
    ],
    "center_back": [
        "Defence","Aerial Defence","Aerial duels per 90","Aerial duels won, %",
        "Defensive duels per 90","Defensive duels won, %","PAdj Interceptions","PAdj Sliding tackles",
        "Passes to final third per 90","Accurate passes %",
        "Work Rate Defensive","Defensive Intensity","Defensive Explosion",
    ],
    "goalkeeper": GK_METRICS + [
        "Passes per 90","Accurate passes, %","Long passes per 90","Accurate long passes, %","Aerial duels per 90","Aerial duels won, %"
    ],
    "general_summary": ["Involvement","Box Threat","Creativity","xG Buildup","Progression","Defence","Discipline","Passing Quality"],
    "defensive_actions": ["Defence","Aerial Defence","PAdj Interceptions","Successful defensive actions per 90","Defensive duels won, %","Shots blocked per 90","Discipline"],
    "playmaking": ["Creativity","Passing Quality","Progression","xG Buildup","Successful attacking actions per 90","Deep completions per 90","Involvement","Key passes per 90"],
    "aerial_duels": ["Aerial Threat","Aerial Defence","Aerial duels per 90","Aerial duels won, %","Head goals per 90","Involvement","Defence"],
    "shooting": ["npxG per 90","npxG per Shot","Finishing","Goal conversion, %","Shots on target, %","G-xG","Box Threat"],
    "counter_attack": ["Progression","Accelerations per 90","Dribbles per 90","Successful dribbles, %","npxG per 90","Finishing","Box Threat"],
    "build_up": ["Passing Quality","Progression","xG Buildup","Deep completions per 90","Involvement","Creativity","Discipline"],
    "crossing": ["Crossing","Accurate crosses, %","Deep completed crosses per 90","xA per 90","Shot assists per 90","Creativity","Passing Quality"],
    "striker": ["npxG per 90","npxG per Shot","Finishing","Poaching","Aerial Threat","Box Threat","Involvement","Touches in box per 90"],
}

NEGATE_METRICS = {
    "Conceded goals per 90": True,
    "xG against per 90": True,
    "Fouls per 90": True,
    "Yellow cards per 90": True,
    "Red cards per 90": True,
}

# ===================== HELPERS =====================
@st.cache_data(show_spinner=False)
def _load_excel(file: io.BytesIO) -> pd.DataFrame:
    try:
        return pd.read_excel(file, sheet_name=0)
    except Exception:
        file.seek(0)
        return pd.read_excel(file)

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _safe_s(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(float).fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)

def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd

# ===================== DERIVED METRICS =====================
def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = set([
        "xG","Penalties taken","Goals","Shots","Touches in box per 90",
        "xA per 90","Shot assists per 90","Key passes per 90","Deep completions per 90",
        "Deep completed crosses per 90","Accurate passes %","Accurate passes, %",
        "Smart passes per 90","Through passes per 90","Passes to penalty area per 90",
        "Accurate smart passes, %","Accurate through passes, %","Accurate passes to penalty area, %",
        "Progressive passes per 90","Progressive runs per 90","Dribbles per 90","Accelerations per 90",
        "Accurate progressive passes, %","Successful dribbles, %",
        "Successful defensive actions per 90","PAdj Interceptions","Shots blocked per 90","PAdj Sliding tackles",
        "Defensive duels per 90","Defensive duels won, %","Aerial duels won, %","Aerial duels per 90",
        "Passes to final third per 90","Accurate passes to final third, %","Forward passes per 90","Accurate forward passes, %",
        "Long passes per 90","Accurate long passes, %","Lateral passes per 90","Accurate lateral passes, %",
        "Back passes per 90","Accurate back passes, %","Passes per 90",
        "Head goals per 90","Goal conversion, %","Received passes per 90",
        "Fouls per 90","Yellow cards per 90","Red cards per 90",
        "Minutes played"
    ])
    _ensure_cols(df, list(needed))

    # npxG & friends
    if "xG" in df and "Penalties taken" in df:
        df["npxG"] = (df["xG"] - (df["Penalties taken"].fillna(0) * 0.81))
    if "Goals" in df and "xG" in df:
        df["G-xG"] = _zscore(df["Goals"] - df["xG"])
    if "npxG" in df and "Minutes played" in df:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["npxG per 90"] = df["npxG"] / (df["Minutes played"].replace(0, np.nan) / 90)
    if "npxG" in df and "Shots" in df:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["npxG per Shot"] = df["npxG"] / df["Shots"].replace(0, np.nan)
    # Box Threat
    if "npxG per 90" in df and "Touches in box per 90" in df:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.log(df["Touches in box per 90"].fillna(0) + 1)
            denom = denom.replace(0, np.nan)
            df["Box Threat"] = _zscore(df["npxG per 90"] / denom)

    # xG Buildup
    xgb_w = {
        "xA per 90": 3.0, "Shot assists per 90": 3.0, "npxG per 90": 2.5, "Key passes per 90": 2.5,
        "Deep completions per 90": 2.5, "Deep completed crosses per 90": 2.0, "Second assists per 90": 1.5,
        "Accurate passes %": 1.0,
    }
    parts = []
    total_w = 0.0
    for col, w in xgb_w.items():
        if col in df.columns:
            s = df[col].fillna(0)
            if "%" in col and s.max() > 1:
                s = s / 100.0
            parts.append(_zscore(s) * w)
            total_w += w
    if parts and total_w > 0:
        df["xG Buildup"] = sum(parts) / total_w

    # Creativity
    vol = {"Smart passes per 90": 1.0, "Through passes per 90": 0.8, "Passes to penalty area per 90": 0.6}
    acc = {"Accurate smart passes, %": 1.0, "Accurate through passes, %": 0.9, "Accurate passes to penalty area, %": 0.8, "Accurate passes %": 0.6}
    vol_scores, acc_scores = [], []
    for c, w in vol.items():
        if c in df.columns: vol_scores.append(_zscore(df[c].fillna(0)) * w)
    for c, w in acc.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if s.max() > 1: s = s/100.0
            acc_scores.append(_zscore(s) * w)
    if vol_scores and acc_scores:
        df["Creativity"] = 0.6 * np.nanmean(vol_scores, axis=0) + 0.4 * np.nanmean(acc_scores, axis=0)

    # Progression
    prog_vol = {"Progressive passes per 90": 1.0, "Progressive runs per 90": 0.9, "Dribbles per 90": 0.7, "Accelerations per 90": 0.6}
    prog_acc = {"Accurate progressive passes, %": 1.0, "Successful dribbles, %": 0.8, "Accurate passes %": 0.3}
    vol_scores, acc_scores = [], []
    for c, w in prog_vol.items():
        if c in df.columns: vol_scores.append(_zscore(df[c].fillna(0)) * w)
    for c, w in prog_acc.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if s.max() > 1: s = s/100.0
            acc_scores.append(_zscore(s) * w)
    if vol_scores and acc_scores:
        df["Progression"] = 0.6 * np.nanmean(vol_scores, axis=0) + 0.4 * np.nanmean(acc_scores, axis=0)

    # Defence
    def_w = {
        "Successful defensive actions per 90": 1.5, "PAdj Interceptions": 1.5, "Shots blocked per 90": 1.0,
        "PAdj Sliding tackles": 1.0, "Defensive duels per 90": 0.5, "Defensive duels won, %": 3.0, "Aerial duels won, %": 1.5,
    }
    parts = []; total_w = 0.0
    for c, w in def_w.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if "%" in c and s.max() > 1: s = s/100.0
            parts.append(_zscore(s) * w); total_w += w
    if parts and total_w > 0:
        df["Defence"] = sum(parts) / total_w

    # Involvement
    inv_w = {
        "Passes per 90": .20, "Received passes per 90": .15, "Touches per 90": .05,
        "Defensive duels per 90": .10, "PAdj Interceptions": .10, "Successful defensive actions per 90": .05,
        "Touches in box per 90": .10, "Offensive duels per 90": .10, "Progressive runs per 90": .05,
        "Aerial duels per 90": .10,
    }
    parts = []
    for c, w in inv_w.items():
        if c in df.columns: parts.append(_zscore(df[c].fillna(0)) * w)
    if parts: df["Involvement"] = sum(parts)

    # Discipline (negative)
    if all(c in df.columns for c in ["Fouls per 90","Yellow cards per 90","Red cards per 90"]):
        penalty = (df["Fouls per 90"].fillna(0)*1.0 + df["Yellow cards per 90"].fillna(0)*2.0 + df["Red cards per 90"].fillna(0)*4.0)
        df["Discipline"] = -_zscore(penalty)

    # Aux: Non-penalty goals per 90
    if all(c in df.columns for c in ["Goals","Penalties taken","Minutes played"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["Non-penalty goals per 90"] = (df["Goals"].fillna(0) - df["Penalties taken"].fillna(0)) / (df["Minutes played"].replace(0, np.nan)/90)

    # Poaching
    poach_w = {"npxG per Shot": .35, "Goal conversion, %": .30, "Touches in box per 90": -.25,
               "Received passes per 90": -.20, "Non-penalty goals per 90": .40}
    parts = []
    for c, w in poach_w.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if "%" in c and s.max() > 1: s = s/100.0
            parts.append(_zscore(s) * w)
    if parts: df["Poaching"] = sum(parts)

    # Finishing
    fin_w = {"Goal conversion, %": .35, "Non-penalty goals per 90": .30, "Shots on target, %": .25, "G-xG": .15, "npxG per Shot": .10}
    parts = []
    for c, w in fin_w.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if "%" in c and s.max() > 1: s = s/100.0
            parts.append(_zscore(s) * w)
    if parts: df["Finishing"] = sum(parts)

    # Aerial Threat
    aer_w = {"Head goals per 90": .35, "Aerial duels per 90": .20, "Aerial duels won, %": .20}
    parts = []
    for c, w in aer_w.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if "%" in c and s.max() > 1: s = s/100.0
            parts.append(_zscore(s) * w)
    if parts: df["Aerial Threat"] = sum(parts)

    # Passing Quality
    pq_pairs = {
        "Passes to final third per 90": ("Accurate passes to final third, %", 0.35),
        "Forward passes per 90": ("Accurate forward passes, %", 0.30),
        "Long passes per 90": ("Accurate long passes, %", 0.15),
        "Lateral passes per 90": ("Accurate lateral passes, %", 0.10),
        "Back passes per 90": ("Accurate back passes, %", 0.05),
        "Passes per 90": ("Accurate passes, %", 0.05),
    }
    parts = []; total_w = 0.0
    for vol_col, (acc_col, w) in pq_pairs.items():
        if vol_col in df.columns and acc_col in df.columns:
            vol_z = _zscore(df[vol_col].fillna(0))
            acc = df[acc_col].fillna(0)
            if acc.max() > 1: acc = acc/100.0
            acc_z = _zscore(acc)
            parts.append((0.3*vol_z + 0.7*acc_z) * w); total_w += w
    if parts and total_w > 0:
        df["Passing Quality"] = sum(parts) / total_w

    # Aerial Defence
    ad_w = {"Aerial duels per 90": .35, "Aerial duels won, %": .40, "PAdj Interceptions": .10, "Shots blocked per 90": .10}
    parts = []
    for c, w in ad_w.items():
        if c in df.columns:
            s = df[c].fillna(0)
            if "%" in c and s.max() > 1: s = s/100.0
            parts.append(_zscore(s) * w)
    if parts: df["Aerial Defence"] = sum(parts)

    # Normalize to [0,100]
    main_derived = [
        "npxG","npxG per 90","npxG per Shot","Box Threat","xG Buildup","Creativity","Progression",
        "Defence","Involvement","Discipline","G-xG","Poaching","Finishing","Aerial Threat","Passing Quality","Aerial Defence"
    ]
    for m in main_derived:
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce")
            if s.notna().sum() > 1:
                lo, hi = np.nanmin(s), np.nanmax(s)
                if np.isfinite(lo) and np.isfinite(hi) and hi != lo:
                    df[m] = (s - lo) / (hi - lo) * 100.0
                else:
                    df[m] = 50.0
            else:
                df[m] = 50.0

    # Invert negatives if present
    for m in ["Conceded goals per 90","xG against per 90","Fouls per 90","Yellow cards per 90","Red cards per 90"]:
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce")
            if s.notna().sum() > 1:
                hi, lo = np.nanmax(s), np.nanmin(s)
                if hi != lo: df[m] = (hi - s) / (hi - lo) * 100.0
                else: df[m] = 50.0
    return df

# ===================== OUR OFF/DEF COMPOSITES =====================
def compute_offensive_production(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    df = _ensure_cols(df, OFFENSIVE_COMPONENTS)
    s = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            s = s + _safe_s(df, col) * float(w)
    return s

def compute_defensive_production(df: pd.DataFrame) -> pd.Series:
    df = _ensure_cols(df, DEF_COLS)
    def_eff = _safe_s(df, "Defensive duels per 90") * (_safe_s(df, "Defensive duels won, %")/100.0)
    aer_eff = _safe_s(df, "Aerial duels per 90") * (_safe_s(df, "Aerial duels won, %")/100.0)
    parts = [
        _safe_s(df, "Successful defensive actions per 90"),
        def_eff, aer_eff,
        _safe_s(df, "PAdj Sliding tackles"),
        _safe_s(df, "PAdj Interceptions"),
        _safe_s(df, "Shots blocked per 90"),
    ]
    total = parts[0]
    for p in parts[1:]:
        total = total + p
    return total

def compute_composite_metrics(df: pd.DataFrame, off_weights: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df = compute_derived_metrics(df)

    df = _ensure_cols(df, list(PHYS_COLS.values()))
    df["Prod_Ofensiva_p90"] = compute_offensive_production(df, off_weights)
    df["Prod_Defensiva_p90"] = compute_defensive_production(df)

    def pick(*cands):
        for c in cands:
            if c in df.columns: return c
        return None
    dist_col = pick("Distance P90","distance_total_m","Total distance per 90","Total Distance per 90 (m)")
    run_col  = pick("Running Distance P90","running_m")
    hi_col   = pick("HI Distance P90","hi_m")
    hsr_cnt  = pick("HSR Count P90","hsr_cnt")
    spr_cnt  = pick("Sprint Count P90","sprint_cnt")
    ex_hsr   = pick("Explosive Acceleration to HSR Count P90","expl_hsr_cnt")
    ex_spr   = pick("Explosive Acceleration to Sprint Count P90","expl_sprint_cnt")

    dist_km = None
    if dist_col:
        series = pd.to_numeric(df[dist_col], errors="coerce").fillna(0)
        dist_km = np.where(series.abs().max() > 2000, series/1000.0, series)

    hi_run_km = None
    if hi_col or run_col:
        s_hi = pd.to_numeric(df[hi_col], errors="coerce").fillna(0) if hi_col else 0.0
        s_run = pd.to_numeric(df[run_col], errors="coerce").fillna(0) if run_col else 0.0
        s_both = s_hi + s_run
        hi_run_km = np.where(s_both.abs().max() > 2000, s_both/1000.0, s_both)

    exp_events = None
    cnts = []
    for c in [hsr_cnt, spr_cnt, ex_hsr, ex_spr]:
        if c: cnts.append(pd.to_numeric(df[c], errors="coerce").fillna(0))
    if cnts: exp_events = sum(cnts)

    def safe_ratio(num, den):
        if num is None or den is None: return np.nan
        den = np.asarray(den); den = np.where(den==0, np.nan, den)
        return num / den

    if dist_km is not None:
        df["Work Rate Offensive"] = safe_ratio(df["Prod_Ofensiva_p90"], dist_km)
        df["Work Rate Defensive"] = safe_ratio(df["Prod_Defensiva_p90"], dist_km)
    if hi_run_km is not None:
        df["Offensive Intensity"] = safe_ratio(df["Prod_Ofensiva_p90"], hi_run_km)
        df["Defensive Intensity"] = safe_ratio(df["Prod_Defensiva_p90"], hi_run_km)
    if exp_events is not None:
        df["Offensive Explosion"] = safe_ratio(df["Prod_Ofensiva_p90"], exp_events)
        df["Defensive Explosion"] = safe_ratio(df["Prod_Defensiva_p90"], exp_events)

    if "Minutes played" not in df.columns:
        raise ValueError("A coluna 'Minutes played' não foi encontrada no arquivo enviado. Certifique-se de manter exatamente esse nome.")
    return df

# ===================== RADAR / UI HELPERS =====================
def _bounds_from_df(df: pd.DataFrame, metrics: list[str]):
    lowers, uppers = [], []
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        if m in NEGATE_METRICS: s = -s
        lo = np.nanpercentile(s, 5); hi = np.nanpercentile(s, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.nanmin(s); hi = np.nanmax(s)
        if lo == hi: hi = lo + 1e-6
        lowers.append(float(lo)); uppers.append(float(hi))
    return lowers, uppers

def _values_for_player(row: pd.Series, metrics: list[str]):
    vals = []
    for m in metrics:
        v = pd.to_numeric(row.get(m, np.nan), errors="coerce")
        if m in NEGATE_METRICS and pd.notna(v): v = -v
        vals.append(float(v) if pd.notna(v) else np.nan)
    return vals

def _player_label(row: pd.Series) -> str:
    name = str(row.get("Player", ""))
    team = str(row.get("Team", "")) if "Team" in row.index and pd.notna(row.get("Team")) else ""
    pos  = str(row.get("Position", "")) if "Position" in row.index and pd.notna(row.get("Position")) else ""
    minutes = None
    for mcol in ["Minutes played","Minutes","minutes","Time played","Minuti","Minutos","Min"]:
        if mcol in row.index and pd.notna(row.get(mcol)):
            try:
                minutes = int(float(row[mcol]))
            except Exception:
                minutes = row[mcol]
            break
    tail = f" — {team}" if team else ""
    if pos: tail += f" | {pos}"
    if minutes is not None: tail += f" | {minutes} min"
    return name + tail

def plot_radar(df: pd.DataFrame, player_a: str, player_b: str | None, metrics: list[str],
               color_a: str, color_b: str = "#E76F51"):
    if not metrics:
        st.warning("Select at least 3 metrics for the radar.")
        return
    metrics = metrics[:16]
    row_a = df[df["Player"] == player_a].iloc[0]
    row_b = df[df["Player"] == player_b].iloc[0] if player_b else None
    lowers, uppers = _bounds_from_df(df, metrics)
    radar = Radar(metrics, lowers, uppers, num_rings=4)

    v_a = _values_for_player(row_a, metrics)
    v_b = _values_for_player(row_b, metrics) if row_b is not None else None

    fig, ax = plt.subplots(figsize=(8, 8))
    radar.setup_axis(ax=ax)
    radar.draw_circles(ax=ax, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
    try:
        radar.spoke(ax=ax, color="#c9c9c9", linestyle="--", alpha=0.18)
    except Exception:
        pass
    radar.draw_radar(v_a, ax=ax, kwargs_radar={"facecolor": color_a+"33", "edgecolor": color_a, "linewidth": 2})
    if v_b is not None:
        radar.draw_radar(v_b, ax=ax, kwargs_radar={"facecolor": color_b+"33", "edgecolor": color_b, "linewidth": 2})
    radar.draw_range_labels(ax=ax, fontsize=9)
    radar.draw_param_labels(ax=ax, fontsize=10)

    title_a = _player_label(row_a)
    title = title_a if row_b is None else f"{title_a} vs {_player_label(row_b)}"
    ax.set_title(title, fontsize=14, pad=20)

    st.pyplot(fig, use_container_width=True)

# ===================== Ranking bars helpers =====================
def _metric_rank_info(dfin: pd.DataFrame, metric: str, player_name: str):
    # Barra baseada em ranking: 1 = melhor => barra cheia (norm = 1.0)
    s = pd.to_numeric(dfin[metric], errors="coerce")
    mask = s.notna()
    s = s[mask]
    d = dfin.loc[mask]
    total = int(s.shape[0]) if s.shape[0] > 0 else 0
    if total == 0:
        return {"rank": None, "total": 0, "value": np.nan, "norm": 0.0, "ascending": False}
    ascending = metric in NEGATE_METRICS  # se True, menor é melhor

    # Rank ordinal (1 = melhor considerando ascending)
    r = s.rank(ascending=ascending, method="min")
    try:
        player_idx = d.index[d["Player"] == player_name]
        if player_idx.empty:
            return {"rank": None, "total": total, "value": np.nan, "norm": 0.0, "ascending": ascending}
        val = float(s.loc[player_idx[0]]) if player_idx[0] in s.index else np.nan
        rk = int(r.loc[player_idx[0]]) if player_idx[0] in r.index and not np.isnan(r.loc[player_idx[0]]) else None
    except Exception:
        val, rk = np.nan, None

    # Converte rank -> [0,1]: 1.0 para rank 1; 0.0 para rank = total
    if rk is None:
        norm = 0.0
    elif total <= 1:
        norm = 1.0
    else:
        norm = 1.0 - (float(rk - 1) / float(total - 1))

    norm = float(np.clip(norm, 0.0, 1.0))
    return {"rank": rk, "total": total, "value": val, "norm": norm, "ascending": ascending}
def render_metric_rank_bars(dfin: pd.DataFrame, player_a: str, metrics: list[str], player_b: str | None = None):
    if not metrics:
        return
    st.markdown("### 📊 Ranking por métrica")
    st.caption("Barra indica desempenho relativo; rótulo mostra a posição no ranking (1 = melhor).")

    def _render_for(player_name: str, header: str):
        st.markdown(f"**{header}:** {player_name}")
        cols_per_row = 3
        for i, m in enumerate(metrics):
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            c = cols[i % cols_per_row]
            with c:
                info = _metric_rank_info(dfin, m, player_name)
                rk, tot, val, norm = info["rank"], info["total"], info["value"], info["norm"]
                label = f"{m} — {rk}/{tot}" if rk is not None else f"{m} — n/a"
                fig, ax = plt.subplots(figsize=(4, 0.6))
                ax.barh([0], [norm])
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(["0%","50%","100%"], fontsize=7)
                ax.set_title(label, fontsize=9, pad=2)
                for spine in ["top","right","left"]:
                    ax.spines[spine].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    _render_for(player_a, "Jogador A")
    if player_b:
        _render_for(player_b, "Jogador B")

# ======= Build a single PNG that includes Radar + Ranking Bars =======
def make_radar_bars_png(df: pd.DataFrame, player_a: str, player_b: str | None, metrics: list[str],
                        color_a: str, color_b: str = "#E76F51") -> io.BytesIO:
    metrics = (metrics or [])[:16]

    lowers, uppers = _bounds_from_df(df, metrics)
    radar = Radar(metrics, lowers, uppers, num_rings=4)

    row_a = df[df["Player"] == player_a].iloc[0]
    v_a = _values_for_player(row_a, metrics)

    row_b = None
    v_b = None
    if player_b:
        row_b = df[df["Player"] == player_b].iloc[0]
        v_b = _values_for_player(row_b, metrics)

    cols_per_row = 3
    rows_per_player = max(1, math.ceil(len(metrics) / cols_per_row))
    bar_blocks = 1 + (1 if player_b else 0)
    total_bar_rows = rows_per_player * bar_blocks

    fig = plt.figure(figsize=(11, 8 + total_bar_rows * 0.9))
    gs = GridSpec(nrows=2 + total_bar_rows, ncols=3, figure=fig)

    # Radar spans first 2 rows
    ax_radar = fig.add_subplot(gs[0:2, :])
    radar.setup_axis(ax=ax_radar)
    radar.draw_circles(ax=ax_radar, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
    try:
        radar.spoke(ax=ax_radar, color="#c9c9c9", linestyle="--", alpha=0.18)
    except Exception:
        pass
    radar.draw_radar(v_a, ax=ax_radar, kwargs_radar={"facecolor": color_a+"33", "edgecolor": color_a, "linewidth": 2})
    if v_b is not None:
        radar.draw_radar(v_b, ax=ax_radar, kwargs_radar={"facecolor": color_b+"33", "edgecolor": color_b, "linewidth": 2})
    radar.draw_range_labels(ax=ax_radar, fontsize=9)
    radar.draw_param_labels(ax=ax_radar, fontsize=10)

    title_a = _player_label(row_a)
    title = title_a if row_b is None else f"{title_a} vs {_player_label(row_b)}"
    ax_radar.set_title(title, fontsize=20, weight="bold", pad=18)

    # Bar blocks (player A then optional player B)
    def _draw_bar_block(start_row: int, player_name: str):
        for i, m in enumerate(metrics):
            r = start_row + (i // cols_per_row)
            c = i % cols_per_row
            ax = fig.add_subplot(gs[2 + r, c])
            info = _metric_rank_info(df, m, player_name)
            rk, tot, norm = info["rank"], info["total"], info["norm"]
            label = f"{m} — {rk}/{tot}" if rk is not None else f"{m} — n/a"
            ax.barh([0], [norm])
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["0%","50%","100%"], fontsize=7)
            ax.set_title(label, fontsize=9, pad=2)
            for spine in ["top","right","left"]:
                ax.spines[spine].set_visible(False)

    _draw_bar_block(start_row=0, player_name=player_a)
    if player_b:
        _draw_bar_block(start_row=rows_per_player, player_name=player_b)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ======= Build an A4 PDF (Radar 3/4 + Bars 1/4) =======
# [removido] função legacy make_radar_bars_pdf_a4


def make_radar_bars_pdf_a4_pro(
    df_all,
    player_a_name,
    player_b_name_or_none,
    metrics_sel,
    color_a,
    color_b,
    *,
    player_photo_bytes=None,
    crest_bytes=None,
    include_bars=True,
    title_left=None,
    title_right=None,
):
    """
    Gera um PDF A4 com cabeçalho + radar e, opcionalmente, barras slim.
    Mantém a assinatura usada no app (args posic.), com extras por keyword.
    Integração: usa seu df_all/metrics_sel reais para montar o Radar e os valores.
    """
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    # Dependências opcionais
    try:
        from PIL import Image
    except Exception:
        Image = None

    # ===== 1) Preparar dados para o Radar (usando seu schema real) =====
    # metrics_sel: lista de dicts OU objetos com name/min/max e nomes de colunas no df_all
    # Vamos suportar os 2 formatos: dict ou objeto com atributos.
    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    params = [_get(m, "name") for m in metrics_sel]
    mins   = [_get(m, "min", 0.0) for m in metrics_sel]
    maxs   = [_get(m, "max", 1.0) for m in metrics_sel]

    # Mapeamento: tentamos primeiro via 'value_map' (se existir), senão buscamos no df_all
    # assumindo que df_all tem linhas por jogador e colunas com o nome da métrica (ou alias em 'col').
    colnames = [(_get(m, "col") or _get(m, "name")) for m in metrics_sel]

    def _values_for_player(pname):
        if not pname or pname == "—":
            return None
        # 1) value_map (por métrica)
        row_vals = []
        for m, col in zip(metrics_sel, colnames):
            vmap = _get(m, "value_map")
            if isinstance(vmap, dict) and pname in vmap:
                row_vals.append(float(vmap[pname]))
                continue
            # 2) df_all lookup
            v = None
            try:
                # Procura linha df_all[df_all["player"] == pname][col]
                if "player" in df_all.columns:
                    sub = df_all.loc[df_all["player"] == pname]
                    if not sub.empty and col in sub.columns:
                        v = sub.iloc[0][col]
                # Fallback: se df_all for pivotado por índice do jogador
                if v is None and pname in getattr(df_all, "index", []):
                    if col in df_all.columns:
                        v = df_all.loc[pname, col]
            except Exception:
                v = None
            # Normaliza para float (NaN vira min)
            try:
                row_vals.append(float(v))
            except Exception:
                row_vals.append(np.nan)
        return np.array(row_vals, dtype=float)

    values_a = _values_for_player(player_a_name)
    values_b = _values_for_player(player_b_name_or_none)

    # ===== 2) Construir figura e grids =====
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # A4
    gs_page = gridspec.GridSpec(nrows=12, ncols=12, figure=fig, hspace=0.00, wspace=0.10)

    # ===== 3) Cabeçalho =====
    if title_left is None:
        title_left = f"{player_a_name}" + (f" vs {player_b_name_or_none}" if player_b_name_or_none and player_b_name_or_none != "—" else "")
    if title_right is None:
        title_right = "Season • Competition • Source"

    ax_head = fig.add_subplot(gs_page[0:2, 0:12])
    ax_head.axis("off")
    def _wrap(text, width=46):
        import textwrap as _tw
        if not text:
            return ""
        return "\n".join(_tw.wrap(str(text), width=width))
    ax_head.text(0.01, 0.62, _wrap(title_left, 46), fontsize=14, fontweight=700, va="center", ha="left")
    ax_head.text(0.99, 0.62, _wrap(title_right, 46), fontsize=11, fontweight=500, va="center", ha="right")

    # Logos/fotos (opcionais)
    if Image and crest_bytes:
        try:
            crest = Image.open(io.BytesIO(crest_bytes))
            ax_head.imshow(crest, extent=(0.90, 0.98, 0.05, 0.55), aspect='auto')
        except Exception:
            pass
    if Image and player_photo_bytes:
        try:
            photo = Image.open(io.BytesIO(player_photo_bytes)).convert("RGB")
            ax_head.imshow(photo, extent=(0.02, 0.18, 0.05, 0.55), aspect='auto')
        except Exception:
            pass

    # ===== 4) Radar =====
    from mplsoccer import Radar
    radar = Radar(params=params, min_range=mins, max_range=maxs, num_rings=4, ring_width=1, center_circle_radius=0.5)
    ax_radar = fig.add_subplot(gs_page[2:9, 0:12])
    radar.setup_axis(ax=ax_radar)
    radar.draw_circles(ax=ax_radar)
    radar.draw_radii_labels(ax=ax_radar, fontsize=9)
    radar.draw_range_labels(ax=ax_radar, fontsize=8)
    radar.draw_param_labels(ax=ax_radar, fontsize=9, offset=1.05)

    # polígonos
    if values_a is not None:
        ra, _ = radar.draw_radar(values_a, ax=ax_radar, kwargs_radar={"alpha":0.20}, kwargs_outline={"linewidth":1.5})
        ra.set_facecolor(color_a); ra.set_edgecolor(color_a)
    if values_b is not None:
        rb, _ = radar.draw_radar(values_b, ax=ax_radar, kwargs_radar={"alpha":0.20}, kwargs_outline={"linewidth":1.5})
        rb.set_facecolor(color_b); rb.set_edgecolor(color_b)

    # legenda
    handles = []
    if values_a is not None:
        handles.append(plt.Line2D([0],[0], marker='o', linestyle='-', linewidth=1.5, markersize=6, color=color_a, label=str(player_a_name)))
    if values_b is not None:
        handles.append(plt.Line2D([0],[0], marker='o', linestyle='-', linewidth=1.5, markersize=6, color=color_b, label=str(player_b_name_or_none)))
    if handles:
        ax_radar.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=len(handles), frameon=False, fontsize=9)

    # ===== 5) Barras slim (opcional) =====
    if include_bars and len(metrics_sel) > 0:
        # Renderizar cards de barras compactas em linhas 9–11, 3 colunas por linha
        start, end = 9, 12
        cols_per_row = 3
        def _val_or_nan(arr, i):
            try:
                return float(arr[i])
            except Exception:
                return float("nan")
        for r in range(start, end):
            for c in range(0, 12, int(12/cols_per_row)):
                idx = (r-start)*cols_per_row + (c//int(12/cols_per_row))
                if idx >= len(metrics_sel):
                    break
                ax = fig.add_subplot(gs_page[r:r+1, c:c+int(12/cols_per_row)])
                ax.set_axisbelow(True)

                name = _get(metrics_sel[idx], "name")
                lo   = _get(metrics_sel[idx], "min", 0.0)
                hi   = _get(metrics_sel[idx], "max", 1.0)
                va   = _val_or_nan(values_a, idx) if values_a is not None else None
                vb   = _val_or_nan(values_b, idx) if values_b is not None else None

                ax.axvspan(lo, hi, ymin=0.35, ymax=0.65, alpha=0.10)
                ax.set_xlim(lo, hi); ax.set_ylim(0, 1)

                if va is not None and not np.isnan(va):
                    ax.hlines(0.55, lo, va, linewidth=6, alpha=0.85, color=color_a)
                    ax.text(va, 0.80, f"{va:.2f}", fontsize=8, ha="center", va="bottom")
                if vb is not None and not np.isnan(vb):
                    ax.hlines(0.45, lo, vb, linewidth=6, alpha=0.85, color=color_b)
                    ax.text(vb, 0.20, f"{vb:.2f}", fontsize=8, ha="center", va="top")

                ax.text(lo, 0.95, name if name is not None else "", fontsize=9, ha="left", va="top", fontweight=600)
                ax.spines[:].set_visible(False)
                ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # ===== 6) Exportação =====
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    buf.seek(0)
    return buf

