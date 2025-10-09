
from __future__ import annotations

import io
import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mplsoccer import Radar


# ========================= PAGE CONFIGURATION AND STYLES =========================
st.set_page_config(
    page_title="Composite Football Metrics and Radar",
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
    .small {font-size: 0.9rem}
    .tight {margin-top: .25rem; margin-bottom: .25rem}
    .muted {color: #5c5f66}
    </style>
    """,
    unsafe_allow_html=True,
)


# ========================= CONSTANTS AND SETTINGS =========================
IDENTITY_COLUMNS = ["Player", "Short Name", "Team", "Position", "Minutes played", "Minutes"]

# Metrics where lower values are better
NEGATIVE_DIRECTION_METRICS = {
    "Conceded goals per 90": True,
    "xG against per 90": True,
    "Fouls per 90": True,
    "Yellow cards per 90": True,
    "Red cards per 90": True,
}

PENALTY_XG_COEFFICIENT = 0.76


# ========================= UTILITY FUNCTIONS =========================
@st.cache_data(show_spinner=False)
def read_excel_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    try:
        return pd.read_excel(uploaded_file, sheet_name="Merged")
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file, sheet_name=0)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)


def ensure_columns_exist(dataframe: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    for name in column_names:
        if name not in dataframe.columns:
            dataframe[name] = np.nan
    return dataframe


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def normalize_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() > 1:
        lo, hi = np.nanmin(s), np.nanmax(s)
        if np.isfinite(lo) and np.isfinite(hi) and hi != lo:
            return (s - lo) / (hi - lo) * 100.0
    return pd.Series(50.0, index=s.index)


# ========================= DERIVED METRICS =========================
def compute_derived_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    minutes_column = "Minutes played" if "Minutes played" in df.columns else ("Minutes" if "Minutes" in df.columns else None)
    if minutes_column is None:
        raise ValueError("A coluna de minutos não foi encontrada. Inclua 'Minutes played' ou 'Minutes'.")

    ensure_columns_exist(df, ["xG", "Penalties taken", "Shots", minutes_column])

    # Penalty xG
    pen_cols = ["Penalty xG", "xG (penalties)", "xG from penalties", "xG pens"]
    pen_xg = None
    for c in pen_cols:
        if c in df.columns:
            pen_xg = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            break
    if pen_xg is None:
        pen_xg = pd.to_numeric(df.get("Penalties taken", 0.0), errors="coerce").fillna(0.0) * PENALTY_XG_COEFFICIENT

    xg = pd.to_numeric(df.get("xG", 0.0), errors="coerce").fillna(0.0)
    df["npxG_raw"] = np.clip(xg - pen_xg, a_min=0.0, a_max=None)

    minutes = pd.to_numeric(df[minutes_column], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["npxG per 90"] = df["npxG_raw"] / (minutes / 90.0)
    df["npxG per 90"] = df["npxG per 90"].replace([np.inf, -np.inf], np.nan)

    shots = pd.to_numeric(df.get("Shots", np.nan), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["npxG per Shot"] = df["npxG_raw"] / shots.replace(0, np.nan)

    # Box Threat
    if "Touches in box per 90" in df.columns:
        tib90 = pd.to_numeric(df["Touches in box per 90"], errors="coerce").fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.log(tib90 + 1.0).replace(0, np.nan)
            df["Box Threat"] = df["npxG per 90"] / denom

    # G-xG (robust numeric coercion; fillna 0 before diff to avoid all-NaN)
    if "Goals" in df.columns and "xG" in df.columns:
        goals_num = pd.to_numeric(df["Goals"], errors="coerce").fillna(0.0)
        xg_num = pd.to_numeric(df["xG"], errors="coerce").fillna(0.0)
        df["G-xG"] = goals_num - xg_num

    # Normalize visuals
    for col in ["npxG per 90", "npxG per Shot", "Box Threat", "G-xG"]:
        if col in df.columns:
            df[col] = normalize_0_100(df[col])

    ensure_columns_exist(df, IDENTITY_COLUMNS)
    return df


# ========================= RADAR HELPERS =========================
def finite_count(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return int(s.notna().sum())


def sanitize_metrics_for_plot(dataframe: pd.DataFrame, metrics: List[str]) -> List[str]:
    """Keep only metrics that have at least 1 finite value across the dataset."""
    valid = []
    for m in metrics:
        if m in dataframe.columns and finite_count(dataframe[m]) >= 1:
            valid.append(m)
    return valid


def bounds_from_dataframe(dataframe: pd.DataFrame, metrics: List[str]) -> Tuple[List[float], List[float]]:
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    for metric in metrics:
        s = pd.to_numeric(dataframe[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if metric in NEGATIVE_DIRECTION_METRICS:
            s = -s
        # if all NaN, give safe default
        if s.notna().sum() == 0:
            low, high = 0.0, 1.0
        else:
            low = np.nanpercentile(s, 5)
            high = np.nanpercentile(s, 95)
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low = np.nanmin(s)
                high = np.nanmax(s)
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = 0.0, 1.0
        if low == high:
            high = low + 1e-6
        lower_bounds.append(float(low))
        upper_bounds.append(float(high))
    return lower_bounds, upper_bounds


def values_for_player(row: pd.Series, metrics: List[str]) -> List[float]:
    values: List[float] = []
    for metric in metrics:
        v = pd.to_numeric(row.get(metric, np.nan), errors="coerce")
        if metric in NEGATIVE_DIRECTION_METRICS and pd.notna(v):
            v = -v
        values.append(float(v) if pd.notna(v) else np.nan)
    return values


def player_label(row: pd.Series) -> str:
    name = str(row.get("Player", ""))
    team = str(row.get("Team", "")) if "Team" in row.index and pd.notna(row.get("Team")) else ""
    position = str(row.get("Position", "")) if "Position" in row.index and pd.notna(row.get("Position")) else ""

    minutes_value = None
    for minutes_col_candidate in ["Minutes played", "Minutes", "Time played", "Min"]:
        if minutes_col_candidate in row.index and pd.notna(row.get(minutes_col_candidate)):
            try:
                minutes_value = int(float(row[minutes_col_candidate]))
            except Exception:
                minutes_value = row[minutes_col_candidate]
            break

    tail = f" — {team}" if team else ""
    if position:
        tail += f" | {position}"
    if minutes_value is not None:
        tail += f" | {minutes_value} min"
    return name + tail


def draw_radar_figure(
    full_dataframe: pd.DataFrame,
    player_a_name: str,
    player_b_name: Optional[str],
    metric_names: List[str],
    color_for_player_a: str,
    color_for_player_b: str = "#E76F51",
) -> plt.Figure:
    if not metric_names:
        raise ValueError("Selecione pelo menos uma métrica para o radar.")

    metric_names = sanitize_metrics_for_plot(full_dataframe, metric_names[:16])
    if not metric_names:
        raise ValueError("Nenhuma métrica com dados válidos para plotar.")

    player_a_row = full_dataframe[full_dataframe["Player"] == player_a_name].iloc[0]
    player_b_row = full_dataframe[full_dataframe["Player"] == player_b_name].iloc[0] if player_b_name else None

    lowers, uppers = bounds_from_dataframe(full_dataframe, metric_names)
    radar = Radar(metric_names, lowers, uppers, num_rings=4)

    values_a = values_for_player(player_a_row, metric_names)
    values_b = values_for_player(player_b_row, metric_names) if player_b_row is not None else None

    fig, ax = plt.subplots(figsize=(8, 8))
    radar.setup_axis(ax=ax)
    radar.draw_circles(ax=ax, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
    try:
        radar.spoke(ax=ax, color="#c9c9c9", linestyle="--", alpha=0.18)
    except Exception:
        pass

    radar.draw_radar(values_a, ax=ax, kwargs_radar={"facecolor": color_for_player_a + "33", "edgecolor": color_for_player_a, "linewidth": 2})
    if values_b is not None:
        radar.draw_radar(values_b, ax=ax, kwargs_radar={"facecolor": color_for_player_b + "33", "edgecolor": color_for_player_b, "linewidth": 2})

    radar.draw_range_labels(ax=ax, fontsize=9)
    radar.draw_param_labels(ax=ax, fontsize=10)

    title_a = player_label(player_a_row)
    title_text = title_a if player_b_row is None else f"{title_a} vs {player_label(player_b_row)}"
    ax.set_title(title_text, fontsize=14, pad=20)

    return fig


# ========================= RANKING BARS =========================
def metric_rank_information(ranking_dataframe: pd.DataFrame, metric_name: str, player_name: str) -> dict:
    s = pd.to_numeric(ranking_dataframe[metric_name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = s.notna()
    s = s[mask]
    d = ranking_dataframe.loc[mask]

    total = int(s.shape[0]) if s.shape[0] > 0 else 0
    if total == 0:
        return {"rank_position": None, "sample_size": 0, "normalized_value": 0.0}

    ascending = metric_name in NEGATIVE_DIRECTION_METRICS
    r = s.rank(ascending=ascending, method="min")

    try:
        idx = d.index[d["Player"] == player_name]
        if idx.empty:
            return {"rank_position": None, "sample_size": total, "normalized_value": 0.0}
        rk = int(r.loc[idx[0]]) if idx[0] in r.index and pd.notna(r.loc[idx[0]]) else None
        val = float(s.loc[idx[0]]) if idx[0] in s.index else np.nan
    except Exception:
        rk, val = None, np.nan

    vmin, vmax = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin or not np.isfinite(val):
        norm = 0.5
    else:
        raw = (val - vmin) / (vmax - vmin)
        norm = 1.0 - raw if ascending else raw
    return {"rank_position": rk, "sample_size": total, "normalized_value": float(np.clip(norm, 0, 1))}


def render_metric_ranking_bars(full_dataframe: pd.DataFrame, player_a_name: str, metric_names: List[str], player_b_name: Optional[str] = None):
    metric_names = sanitize_metrics_for_plot(full_dataframe, metric_names)
    if not metric_names:
        st.info("Sem métricas com dados válidos para exibir as barras.")
        return

    st.markdown("### 📊 Ranking por métrica")
    st.caption("Cada barra indica desempenho relativo (0–100%). O rótulo mostra a posição no ranking (1 = melhor).")

    def render_for(name: str, header: str):
        st.markdown(f"**{header}:** {name}")
        cols_per_row = 3
        for i, m in enumerate(metric_names):
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            c = cols[i % cols_per_row]
            with c:
                info = metric_rank_information(full_dataframe, m, name)
                rk, tot, norm = info["rank_position"], info["sample_size"], info["normalized_value"]
                label = f"{m} — {rk}/{tot}" if rk is not None else f"{m} — n/a"
                fig, ax = plt.subplots(figsize=(4, 0.45))
                ax.barh([0], [norm])
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
                ax.set_title(label, fontsize=9, pad=2)
                for s in ["top", "right", "left"]:
                    ax.spines[s].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    render_for(player_a_name, "Jogador A")
    if player_b_name:
        render_for(player_b_name, "Jogador B")


# ========================= EXPORTS =========================
def build_png_with_radar_and_bars(full_dataframe: pd.DataFrame, player_a_name: str, player_b_name: Optional[str], metric_names: List[str], color_a: str, color_b: str) -> io.BytesIO:
    metric_names = sanitize_metrics_for_plot(full_dataframe, (metric_names or [])[:16])
    cols_per_row = 3
    rows_per_player = max(1, math.ceil(len(metric_names) / cols_per_row))
    bar_blocks = 1 + (1 if player_b_name else 0)
    total_bar_rows = rows_per_player * bar_blocks

    fig = plt.figure(figsize=(11, 8 + total_bar_rows * 0.7))
    gs = GridSpec(nrows=2 + total_bar_rows, ncols=3, figure=fig)

    # Radar
    ax_radar = fig.add_subplot(gs[0:2, :])
    lowers, uppers = bounds_from_dataframe(full_dataframe, metric_names)
    radar = Radar(metric_names, lowers, uppers, num_rings=4)

    row_a = full_dataframe[full_dataframe["Player"] == player_a_name].iloc[0]
    row_b = full_dataframe[full_dataframe["Player"] == player_b_name].iloc[0] if player_b_name else None
    v_a = values_for_player(row_a, metric_names)
    v_b = values_for_player(row_b, metric_names) if row_b is not None else None

    radar.setup_axis(ax=ax_radar)
    radar.draw_circles(ax=ax_radar, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
    try:
        radar.spoke(ax=ax_radar, color="#c9c9c9", linestyle="--", alpha=0.18)
    except Exception:
        pass
    radar.draw_radar(v_a, ax=ax_radar, kwargs_radar={"facecolor": color_a + "33", "edgecolor": color_a, "linewidth": 2})
    if v_b is not None:
        radar.draw_radar(v_b, ax=ax_radar, kwargs_radar={"facecolor": color_b + "33", "edgecolor": color_b, "linewidth": 2})
    radar.draw_range_labels(ax=ax_radar, fontsize=9)
    radar.draw_param_labels(ax=ax_radar, fontsize=10)
    title = player_label(row_a) if row_b is None else f"{player_label(row_a)} vs {player_label(row_b)}"
    ax_radar.set_title(title, fontsize=14, pad=16)

    # Bars
    def draw_bar_block(start_row: int, name: str):
        for i, m in enumerate(metric_names):
            r = start_row + (i // cols_per_row)
            c = i % cols_per_row
            ax = fig.add_subplot(gs[2 + r, c])
            info = metric_rank_information(full_dataframe, m, name)
            rk, tot, norm = info["rank_position"], info["sample_size"], info["normalized_value"]
            label = f"{m} — {rk}/{tot}" if rk is not None else f"{m} — n/a"
            ax.barh([0], [norm]); ax.set_xlim(0, 1); ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
            ax.set_title(label, fontsize=9, pad=2)
            for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)

    draw_bar_block(0, player_a_name)
    if player_b_name: draw_bar_block(rows_per_player, player_b_name)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_two_page_pdf_with_radar_and_bars(full_dataframe: pd.DataFrame, player_a_name: str, player_b_name: Optional[str], metric_names: List[str], color_a: str, color_b: str) -> io.BytesIO:
    metric_names = sanitize_metrics_for_plot(full_dataframe, (metric_names or [])[:16])
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: Radar (portrait)
        fig1 = draw_radar_figure(full_dataframe, player_a_name, player_b_name, metric_names, color_a, color_b)
        pdf.savefig(fig1, dpi=300, bbox_inches="tight"); plt.close(fig1)

        # Page 2: Bars (landscape, thin)
        rows = len(metric_names)
        y = np.arange(rows)[::-1]
        fig2 = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax_title = fig2.add_subplot(111); ax_title.set_title("Ranking por métrica", fontsize=12, pad=10); ax_title.axis("off")

        left, bottom, width, height_ax = 0.08, 0.08, 0.84, 0.84
        axbars = fig2.add_axes([left, bottom, width, height_ax])

        def labels_and_values(name: str):
            labels, norms = [], []
            for m in metric_names:
                info = metric_rank_information(full_dataframe, m, name)
                rk, tot, norm = info["rank_position"], info["sample_size"], info["normalized_value"]
                labels.append(f"{m} — {rk}/{tot}" if rk is not None else f"{m} — n/a")
                norms.append(norm)
            return labels, norms

        labels_a, norms_a = labels_and_values(player_a_name)
        axbars.barh(y, norms_a, height=0.18, label=f"{player_a_name}")
        if player_b_name:
            labels_b, norms_b = labels_and_values(player_b_name)
            axbars.barh(y - 0.2, norms_b, height=0.18, label=f"{player_b_name}")

        axbars.set_xlim(0, 1); axbars.set_yticks(y); axbars.set_yticklabels(labels_a, fontsize=8)
        axbars.set_xticks([0, 0.5, 1]); axbars.set_xticklabels(["0%", "50%", "100%"], fontsize=8)
        for s in ["top", "right", "left"]: axbars.spines[s].set_visible(False)
        axbars.legend(loc="lower right", fontsize=8)

        pdf.savefig(fig2, dpi=300, bbox_inches="tight"); plt.close(fig2)

    buf.seek(0)
    return buf


# ========================= SIDEBAR / DATA =========================
st.sidebar.header("⚙️ Configurações")
uploaded_excel_file = st.sidebar.file_uploader("Envie seu Excel (ex.: SCWY Allsvenskan 2025.xlsx)", type=["xlsx"])
top_n_items_to_show = st.sidebar.slider("Top N no ranking", 5, 50, 10, 1)
position_filter_pattern = st.sidebar.text_input("Filtrar por posição (regex)", value="")
team_filter_exact = st.sidebar.text_input("Filtrar por time (igualdade exata)", value="")
demo_mode_enabled = st.sidebar.checkbox("Modo demo (dados de exemplo)", value=False)
minimum_minutes_for_rankings = st.sidebar.number_input("Mínimo de minutos (apenas para exibição no Ranking)", min_value=0, value=0, step=90, help="Este filtro NÃO afeta os percentis ou o radar; apenas quem aparece nas tabelas de ranking.")

st.title("⚽ Composite Football Metrics and Radar")

if demo_mode_enabled:
    st.warning("Modo demo ativado — dados de exemplo carregados.")
    df_loaded = pd.DataFrame({
        "Player": ["Player A", "Player B", "Player C", "Player D"],
        "Team": ["X", "X", "Y", "Z"],
        "Position": ["CF", "RW", "DMF", "CB"],
        "Minutes played": [900, 880, 910, 1200],
        "xG": [6.0, 4.8, 1.2, 2.0],
        "Penalties taken": [1, 0, 0, 2],
        "Shots": [20, 18, 9, 15],
        "Touches in box per 90": [3.5, 4.0, 1.2, 0.8],
        "Goals": [5, 4, 1, 3],
    })
elif uploaded_excel_file is not None:
    df_loaded = read_excel_file(uploaded_excel_file)
else:
    st.info("Envie seu Excel no painel à esquerda ou habilite o Modo demo.")
    st.stop()

if df_loaded is None or df_loaded.empty:
    st.error("Não foi possível carregar dados.")
    st.stop()

with st.spinner("Processando métricas derivadas..."):
    df_all = compute_derived_metrics(df_loaded)

# Rankings view (minutes filter only affects display)
if "Minutes played" in df_all.columns:
    minutes_col = "Minutes played"
elif "Minutes" in df_all.columns:
    minutes_col = "Minutes"
else:
    st.error("A base não possui 'Minutes played' nem 'Minutes'."); st.stop()

df_view = df_all[df_all[minutes_col].fillna(0) >= int(minimum_minutes_for_rankings)].copy()
st.caption(f"O filtro de minutos afeta apenas as **tabelas de Ranking** (mostrando {df_view.shape[0]} de {df_all.shape[0]} jogadores). Cálculos de radar e percentis usam o **dataset completo**.")

# KPIs
st.subheader("Panorama")
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f"<div class='metric-card'><div class='subtle small'>Jogadores</div><h3>{df_all.shape[0]}</h3></div>", unsafe_allow_html=True)
with c2: st.markdown(f"<div class='metric-card'><div class='subtle small'>Times</div><h3>{df_all['Team'].nunique() if 'Team' in df_all.columns else '—'}</h3></div>", unsafe_allow_html=True)
with c3: st.markdown(f"<div class='metric-card'><div class='subtle small'>Posições</div><h3>{df_all['Position'].nunique() if 'Position' in df_all.columns else '—'}</h3></div>", unsafe_allow_html=True)
with c4: st.markdown(f"<div class='metric-card'><div class='subtle small'>Colunas</div><h3>{df_all.shape[1]}</h3></div>", unsafe_allow_html=True)

# Rankings
st.markdown("<div class='section'>Rankings</div>", unsafe_allow_html=True)
numeric_metric_options = sorted([c for c in df_all.columns if c not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(df_all[c])])
default_index = numeric_metric_options.index("npxG per 90") if "npxG per 90" in numeric_metric_options else 0
selected_metric_for_ranking = st.selectbox("Escolha a métrica para o ranking", numeric_metric_options, index=default_index)

def render_leaderboard_table(metric_name: str):
    d = df_view.copy()
    if d.empty:
        st.info("Nenhum jogador atende ao filtro de minutos para exibição no ranking."); return
    if metric_name not in d.columns:
        st.warning(f"Métrica {metric_name} não encontrada no dataset."); return

    if team_filter_exact:
        d = d[d["Team"].astype(str) == team_filter_exact]
    if position_filter_pattern:
        d = d[d["Position"].astype(str).str.contains(position_filter_pattern, na=False)]

    d[metric_name] = pd.to_numeric(d[metric_name], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_name])
    if d.empty:
        st.info("Sem valores válidos após limpeza para esta métrica."); return

    asc = metric_name in NEGATIVE_DIRECTION_METRICS
    d = d.sort_values(metric_name, ascending=asc, kind="mergesort")

    id_cols_present = [c for c in IDENTITY_COLUMNS if c in d.columns]
    st.dataframe(d.head(top_n_items_to_show)[id_cols_present + [metric_name]], use_container_width=True)

render_leaderboard_table(selected_metric_for_ranking)

# Radar + Bars
st.markdown("<div class='section'>Radar</div>", unsafe_allow_html=True)
players = sorted(df_all["Player"].dropna().unique().tolist()) if "Player" in df_all.columns else []
player_a = st.selectbox("Jogador A", players, index=0 if players else None)
player_b_choice = st.selectbox("Jogador B (opcional)", ["—"] + players, index=0)
player_b = None if player_b_choice == "—" else player_b_choice

metrics_all = [c for c in df_all.columns if c not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(df_all[c])]
metrics_selected = st.multiselect("Métricas (máx 16)", sorted(metrics_all), default=["npxG per 90", "npxG per Shot", "Box Threat", "G-xG"])[:16]
color_a = st.color_picker("Cor A", "#2A9D8F")
color_b = st.color_picker("Cor B", "#E76F51")

if player_a and metrics_selected:
    try:
        fig = draw_radar_figure(df_all, player_a, player_b, metrics_selected, color_a, color_b)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        render_metric_ranking_bars(df_all, player_a, metrics_selected, player_b)
    except Exception as e:
        st.error("Não foi possível gerar o radar com as métricas selecionadas (provavelmente por falta de dados válidos). Ajuste as métricas e tente novamente.")
        st.exception(e)

    png_buf = build_png_with_radar_and_bars(df_all, player_a, player_b, metrics_selected, color_a, color_b)
    st.download_button("⬇️ PNG: Radar + Barras (1 página)", data=png_buf.getvalue(), file_name="radar_barras.png", mime="image/png")

    pdf_buf = build_two_page_pdf_with_radar_and_bars(df_all, player_a, player_b, metrics_selected, color_a, color_b)
    st.download_button("⬇️ PDF (2 páginas): Radar + Barras (paisagem)", data=pdf_buf.getvalue(), file_name="radar_barras.pdf", mime="application/pdf")
