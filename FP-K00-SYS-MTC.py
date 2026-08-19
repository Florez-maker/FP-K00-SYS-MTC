import warnings
warnings.filterwarnings("ignore")

import io
import os
import re
import pathlib

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import streamlit as st
from PIL import Image
import unidecode

# ════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════

st.set_page_config(
    page_title="Dashboard Principal — FarmPrecision",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary":  "#1b60a7",
    "success":  "#2ca02c",
    "danger":   "#d62728",
    "warning":  "#F1C40F",
    "info":     "#17becf",
    "bg":       "#F0F4F8",
    "Q1": "brown",
    "Q2": "orange",
    "Q3": "green",
    "Q4": "dodgerblue",
}

BMA_MAP = {
    "BAJO": 0,
    "MEDIO": 1,
    "ALTO": 2,
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

TEXT_COLS = {
    "finca",
    "lote",
    "departamento",
    "variedad",
    "material",
    "textura_es",
    "textura_en",
    "prof",
    "ifc_clase",
    "tipo_de_analise_foliar",
    "tipo_analisis_foliar",
    "fertilizantes",
    "produto_comercial",
    "producto_comercial",
}

CLIMA_ALIASES = {
    "pp_anual": [
        "pp anual",
        "pp_anual",
        "precipitacion anual",
        "precipitacion_anual",
        "precipitación anual",
        "precipitacion",
    ],
    "temp_promedio": [
        "temp promedio",
        "temp_promedio",
        "temperatura promedio",
        "temperatura_promedio",
    ],
    "humedad_relativa_promedio": [
        "humedad relativa promedio",
        "humedad_relativa_promedio",
        "hr promedio",
        "hr_prom",
        "humedad_relativa",
    ],
    "brillo_solar_promedio": [
        "brillo solar promedio",
        "brillo_solar_promedio",
        "brillo solar",
        "brillo_solar",
    ],
}

ANOMALIA_ALIASES = {
    "fratura_copa": [
        "fratura de copa",
        "fratura_copa",
        "fractura de copa",
        "fractura_copa",
    ],
    "amarelecimento_fatal": [
        "amarelecimento fatal",
        "amarelecimento_fatal",
    ],
    "anel_vermelho": [
        "anel vermelho",
        "anel_vermelho",
        "anillo rojo",
        "anillo_rojo",
    ],
    "opsiphanes_invirae": [
        "opsiphanes invirae",
        "opsiphanes_invirae",
    ],
}

DOSIS_FERTILIZANTE_ALIASES = {
    "dosagem_kg_planta": [
        "dosagem (kg/planta)",
        "dosagem_kg_planta",
        "dosagem kg planta",
        "dosis (kg/planta)",
        "dosis_kg_planta",
        "dosis kg planta",
    ],
}

FOLIAR_ELEMENTS = {
    "n": "fol_n",
    "p": "fol_p",
    "k": "fol_k",
    "ca": "fol_ca",
    "mg": "fol_mg",
    "cl": "fol_cl",
    "s": "fol_s",
    "b": "fol_b",
    "cu": "fol_cu",
    "fe": "fol_fe",
    "mn": "fol_mn",
    "zn": "fol_zn",
}

FOLIAR_ALIASES_RAW = {
    "fol_n": [
        "N.1", "N_f", "N_F", "N_fol", "N_foliar",
        "n_f", "n_fol", "n_foliar", "fol_n",
    ],
    "fol_p": [
        "P.1", "P_f", "P_F", "P_fol", "P_foliar",
        "p_f", "p_fol", "p_foliar", "fol_p",
    ],
    "fol_k": [
        "K.1", "K_f", "K_F", "K_fol", "K_foliar",
        "k_f", "k_fol", "k_foliar", "fol_k",
    ],
    "fol_ca": [
        "Ca.1", "Ca_f", "Ca_F", "Ca_fol", "Ca_foliar",
        "ca_f", "ca_fol", "ca_foliar", "fol_ca",
    ],
    "fol_mg": [
        "Mg.1", "Mg_f", "Mg_F", "Mg_fol", "Mg_foliar",
        "mg_f", "mg_fol", "mg_foliar", "fol_mg",
    ],
    "fol_cl": [
        "Cl.1", "Cl_f", "Cl_F", "Cl_fol", "Cl_foliar",
        "cl_f", "cl_fol", "cl_foliar", "fol_cl",
    ],
    "fol_s": [
        "S.1", "S_f", "S_F", "S_fol", "S_foliar",
        "s_f", "s_fol", "s_foliar", "fol_s",
    ],
    "fol_b": [
        "B.1", "B_f", "B_F", "B_fol", "B_foliar",
        "b_f", "b_fol", "b_foliar", "fol_b",
    ],
    "fol_cu": [
        "Cu.1", "Cu_f", "Cu_F", "Cu_fol", "Cu_foliar",
        "cu_f", "cu_fol", "cu_foliar", "fol_cu",
    ],
    "fol_fe": [
        "Fe.1", "Fe_f", "Fe_F", "Fe_fol", "Fe_foliar",
        "fe_f", "fe_fol", "fe_foliar", "fol_fe",
    ],
    "fol_mn": [
        "Mn.1", "Mn_f", "Mn_F", "Mn_fol", "Mn_foliar",
        "mn_f", "mn_fol", "mn_foliar", "fol_mn",
    ],
    "fol_zn": [
        "Zn.1", "Zn_f", "Zn_F", "Zn_fol", "Zn_foliar",
        "zn_f", "zn_fol", "zn_foliar", "fol_zn",
    ],
}

DIMENSIONES_CORR = {
    # ═══════════════════════════════════════════════════════════════════════
    # 1. SUELO: REACCIÓN, SALINIDAD, ACIDEZ Y CIC
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Reacción, salinidad, acidez y CIC": [
        "ph",
        "cea",
        "ce",
        "cic",
        "cice",
        "acid_int",
        "al",
        "sat_al",
        "sat_na",
        "psi",
        "ras",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SUELO: BASES INTERCAMBIABLES Y MATERIA ORGÁNICA
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Bases intercambiables y materia orgánica": [
        "mo",
        "p",
        "p_meh",
        "p_res",
        "p_rem",
        "p_total",
        "ca",
        "mg",
        "k",
        "na",
        "sat_ca",
        "sat_mg",
        "sat_k",
        "sat_na",
        "sat_bases",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 3. SUELO: RELACIONES CATIÓNICAS
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Relaciones catiónicas": [
        "k_na",
        "mg_k",
        "ca_k",
        "ca_mg",
        "ca_mg_k",
        "ca_mg__k",
        "rel_al_bases",
        "rel_na_bases",
        "bases_suma",
        "bases_ca_mg_k",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 4. SUELO: SATURACIONES Y PARTICIPACIÓN EN CIC
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Saturaciones y porcentajes de CIC": [
        "sat_ca",
        "sat_mg",
        "sat_k",
        "sat_na",
        "sat_al",
        "ca_pct_cic",
        "mg_pct_cic",
        "k_pct_cic",
        "na_pct_cic",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 5. SUELO: MICRONUTRIENTES
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Micronutrientes": [
        "b",
        "cu",
        "fe",
        "mn",
        "zn",
        "s",
        "fe_mn",
        "micro_balance_b_zn",
        "micro_balance_fe_zn",
        "micro_balance_cu_zn",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 6. SUELO: TEXTURA
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Textura": [
        "a",
        "l",
        "ar",
        "textura_fina",
        "rel_arena_arcilla",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 7. ÍNDICES AGRONÓMICOS DEL SUELO
    # ═══════════════════════════════════════════════════════════════════════
    "Suelo — Índices agronómicos": [
        "indice_acidez",
        "indice_bases",
        "indice_sodicidad",
        "indice_micros",
        "indice_fertilidad_quimica",
        "ifc",
        "bases_suma",
        "bases_ca_mg_k",
        "rel_al_bases",
        "rel_na_bases",
    ],

    # ═══════════════════════════════════════════════════════════════════════
    # 8. INVENTARIO, DENSIDAD Y ESTRUCTURA DEL CULTIVO
    # ═══════════════════════════════════════════════════════════════════════
    "Manejo — Inventario y densidad": [
        "edad",
        "n_palmas",
        "densidad",
        "area",
        "da_min",
        "da_avg",
        "da_max",
    ],

    "Fertilización — Oferta nutricional": [
        "oferta_n_kg_ha_avg",
        "oferta_p_kg_ha_avg",
        "oferta_k_kg_ha_avg",
        "oferta_ca_kg_ha_avg",
        "oferta_mg_kg_ha_avg",
        "oferta_s_kg_ha_avg",
        "oferta_b_kg_ha_avg",
        "oferta_zn_kg_ha_avg",
        "oferta_cu_kg_ha_avg",
        "oferta_fe_kg_ha_avg",
        "oferta_mn_kg_ha_avg",

        "oferta_n_kg_ha_min",
        "oferta_p_kg_ha_min",
        "oferta_k_kg_ha_min",
        "oferta_ca_kg_ha_min",
        "oferta_mg_kg_ha_min",
        "oferta_s_kg_ha_min",
        "oferta_b_kg_ha_min",
        "oferta_zn_kg_ha_min",
        "oferta_cu_kg_ha_min",
        "oferta_fe_kg_ha_min",
        "oferta_mn_kg_ha_min",

        "oferta_n_kg_ha_max",
        "oferta_p_kg_ha_max",
        "oferta_k_kg_ha_max",
        "oferta_ca_kg_ha_max",
        "oferta_mg_kg_ha_max",
        "oferta_s_kg_ha_max",
        "oferta_b_kg_ha_max",
        "oferta_zn_kg_ha_max",
        "oferta_cu_kg_ha_max",
        "oferta_fe_kg_ha_max",
        "oferta_mn_kg_ha_max",

        "kgn_ton",
        "kgp_ton",
        "kgk_ton",
    ],

    "Fertilización — Dosis y aplicación": [
        "kg_ha_n",
        "kg_n_ha",
        "kg_ha_p",
        "kg_p_ha",
        "kg_ha_k",
        "kg_k_ha",
        "kg_ha_ca",
        "kg_ca_ha",
        "kg_ha_mg",
        "kg_mg_ha",
        "kg_ha_s",
        "kg_s_ha",
        "kg_ha_b",
        "kg_b_ha",
        "kg_ha_zn",
        "kg_zn_ha",
        "kg_ha_cu",
        "kg_cu_ha",
        "kg_ha_fe",
        "kg_fe_ha",
        "kg_ha_mn",
        "kg_mn_ha",
        "kg_palma",
        "dosagem_kg_planta",
    ],

    "Suelo — Distancia y cumplimiento de rango óptimo": [
        "dist_opt_ph",
        "dist_opt_cea",
        "dist_opt_ce",
        "dist_opt_mo",
        "dist_opt_p",
        "dist_opt_ca",
        "dist_opt_mg",
        "dist_opt_k",
        "dist_opt_na",
        "dist_opt_al",
        "dist_opt_cic",
        "dist_opt_cice",
        "dist_opt_b",
        "dist_opt_cu",
        "dist_opt_fe",
        "dist_opt_mn",
        "dist_opt_zn",
        "dist_opt_s",

        "score_ph",
        "score_cea",
        "score_opt_ph",
        "score_opt_cea",
        "score_opt_ce",
        "score_opt_mo",
        "score_opt_p",
        "score_opt_ca",
        "score_opt_mg",
        "score_opt_k",
        "score_opt_na",
        "score_opt_al",
        "score_opt_cic",
        "score_opt_cice",
        "score_opt_b",
        "score_opt_cu",
        "score_opt_fe",
        "score_opt_mn",
        "score_opt_zn",
        "score_opt_s",
    ],

    "Foliar — Elementos nutricionales": [
        "fol_n",
        "fol_p",
        "fol_k",
        "fol_ca",
        "fol_mg",
        "fol_cl",
        "fol_s",
        "fol_b",
        "fol_cu",
        "fol_fe",
        "fol_mn",
        "fol_zn",
    ],

    "Clima — Variables climatológicas": [
        "pp_anual",
        "temp_promedio",
        "humedad_relativa_promedio",
        "brillo_solar_promedio",
    ],

    "Sanidad — Anomalías y plagas": [
        "fratura_copa",
        "amarelecimento_fatal",
        "anel_vermelho",
        "opsiphanes_invirae",
    ],

    "Suelo — Clasificación BMA ordinal": [
        "bma_ph_ord",
        "bma_cea_ord",
        "bma_ce_ord",
        "bma_mo_ord",
        "bma_p_ord",
        "bma_ca_ord",
        "bma_mg_ord",
        "bma_k_ord",
        "bma_na_ord",
        "bma_cic_ord",
        "bma_cice_ord",
        "bma_b_ord",
        "bma_cu_ord",
        "bma_fe_ord",
        "bma_mn_ord",
        "bma_zn_ord",
        "bma_s_ord",
    ],
}

def sanitize_key(s):
    return (str(s).replace(" ", "_").replace("/", "_").replace("-", "_")
                  .replace(".", "_").replace("(", "").replace(")", ""))

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem; border-radius: 12px;
        color: white; margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
    .kpi-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }
    [data-testid="stSidebar"] { background: #F0F4F8; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 1. CARGA Y NORMALIZACIÓN DE DATOS
# ──────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

def normalize_colname(col: str) -> str:
    col = unidecode.unidecode(str(col).strip()).lower()
    col = col.replace("%", "pct")
    col = col.replace("/", "_")
    col = col.replace("-", "_")
    col = col.replace("+", "_")
    col = col.replace(".", "_")
    col = re.sub(r"[()\[\]{}]", "", col)
    col = re.sub(r"[^a-z0-9_]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

def normalizar_texto(value):
    try:
        if pd.isna(value):
            return np.nan
    except (TypeError, ValueError):
        return np.nan

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "nat", "<​na>"}:
        return np.nan

    return unidecode.unidecode(value).upper()

def safe_slug(value) -> str:
    value = normalizar_texto(value)

    if pd.isna(value):
        value = "SIN_DATO"

    value = re.sub(r"[^A-Z0-9_]+", "_", str(value))
    value = re.sub(r"_+", "_", value).strip("_")

    return value if value else "SIN_DATO"

def file_exists(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _resolve_col(df_cols, target):
    target = str(target).lower()
    exact = [c for c in df_cols if c.lower() == target]
    if exact:
        return exact[0]
    partial = [c for c in df_cols if target in c.lower()]
    return partial[0] if partial else None

def _forzar_numericas_suelo(df):
    all_cols = {c for cols in DIMENSIONES_CORR.values() for c in cols}
    for c in all_cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    return df

def read_table(path: str, sheet_name=None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        if sheet_name is not None:
            return pd.read_excel(path, sheet_name=sheet_name)
        return pd.read_excel(path)

    return pd.read_csv(path, encoding="utf-8-sig")

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def _find_column_case_insensitive(columns, candidate):
    candidate_norm = normalize_colname(candidate)

    for col in columns:
        if normalize_colname(col) == candidate_norm:
            return col

    return None

def detectar_foliar_por_patron(raw_col: str):
    raw = str(raw_col).strip()
    norm = normalize_colname(raw)

    if norm in FOLIAR_ELEMENTS.values():
        return norm

    match = re.fullmatch(r"fol_([a-z]+)", norm)
    if match:
        element = match.group(1)
        return FOLIAR_ELEMENTS.get(element)

    match = re.fullmatch(r"([a-z]+)_f", norm)
    if match:
        element = match.group(1)
        return FOLIAR_ELEMENTS.get(element)

    match = re.fullmatch(r"([a-z]+)_(fol|foliar)", norm)
    if match:
        element = match.group(1)
        return FOLIAR_ELEMENTS.get(element)

    match = re.fullmatch(r"\s*([A-Za-z]+)\.1\s*", raw)
    if match:
        element = normalize_colname(match.group(1))
        return FOLIAR_ELEMENTS.get(element)

    return None

def renombrar_foliares_raw(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    rename_map = {}
    provenance = {}

    for canonical, aliases in FOLIAR_ALIASES_RAW.items():
        for alias in aliases:
            match = _find_column_case_insensitive(df.columns, alias)

            if match is not None and match not in rename_map:
                rename_map[match] = canonical
                provenance[canonical] = {
                    "origen": match,
                    "metodo": "alias_explicito",
                }
                break

    for col in df.columns:
        if col in rename_map:
            continue

        canonical = detectar_foliar_por_patron(col)

        if canonical is None:
            continue

        if canonical in rename_map.values():
            print(
                f"  ⚠️ Foliar duplicado detectado: '{col}' no se renombra porque "
                f"'{canonical}' ya fue asignado."
            )
            continue

        rename_map[col] = canonical
        provenance[canonical] = {
            "origen": col,
            "metodo": "patron_seguro",
        }

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, provenance

def resolve_aliases(df: pd.DataFrame, alias_map: dict) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    found = []

    columns_by_normalized_name = {
        normalize_colname(col): col
        for col in df.columns
    }

    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            found.append(canonical)
            continue

        for alias in aliases:
            alias_norm = normalize_colname(alias)

            if alias_norm in columns_by_normalized_name:
                original_col = columns_by_normalized_name[alias_norm]

                df = df.rename(columns={original_col: canonical})
                found.append(canonical)
                break

    return df, found

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    df.columns = [normalize_colname(c) for c in df.columns]

    duplicated = df.columns[df.columns.duplicated()].unique().tolist()

    if duplicated:
        print(f"  ⚠️ Columnas duplicadas tras normalizar: {duplicated}")
        print("  ⚠️ Se conservará la primera ocurrencia de cada columna.")

    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_texto)

    return df

def cargar_dataset_consolidado(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:

    is_csv = isinstance(file_name, str) and file_name.lower().endswith(".csv")

    if is_csv:
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", dtype=object)
        except Exception:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=",", dtype=object)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    df_raw.columns = df_raw.columns.str.strip().str.lower()
    df_raw, foliar_provenance = renombrar_foliares_raw(df_raw)

    df = standardize_df(df_raw)

    df, clima_detectado = resolve_aliases(df, CLIMA_ALIASES)
    df, anomalias_detectadas = resolve_aliases(df, ANOMALIA_ALIASES)
    df, dosis_detectada = resolve_aliases(df, DOSIS_FERTILIZANTE_ALIASES)

    categorical_aliases = {
        "tipo_de_analise_foliar": [
            "tipo_de_analise_foliar",
            "tipo_de_analisis_foliar",
            "tipo analise foliar",
            "tipo analisis foliar",
        ],
        "fertilizantes": [
            "fertilizantes",
            "fertilizante",
        ],
        "produto_comercial": [
            "produto_comercial",
            "producto_comercial",
            "produto comercial",
            "producto comercial",
        ],
    }

    df, _ = resolve_aliases(df, categorical_aliases)

    bma_cols = [c for c in df.columns if c.startswith("bma_")]

    skip_numeric_conversion = (
        TEXT_COLS
        | set(bma_cols)
        | {c for c in df.columns if c.startswith("fol_")}
    )

    for col in df.columns:
        if col not in skip_numeric_conversion:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [c for c in df.columns if c.startswith("fol_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  ✓ Consolidado estandarizado: {df.shape}")

    foliares_finales = [c for c in df.columns if c.startswith("fol_")]

    print("\n[FOLIARES] Variables detectadas:")
    if foliares_finales:
        for col in foliares_finales:
            info = foliar_provenance.get(col, {})
            print(
                f"  ✓ {col:10s} <- {info.get('origen', 'ya_estandarizada')} "
                f"| método={info.get('metodo', 'preexistente')}"
            )
    else:
        print("  ⚠️ No se detectaron columnas foliares.")

    print("\n[CLIMA] Variables detectadas:")
    print(f"  {clima_detectado if clima_detectado else '⚠️ Ninguna'}")

    print("\n[ANOMALÍAS / PLAGAS] Variables detectadas:")
    print(f"  {anomalias_detectadas if anomalias_detectadas else '⚠️ Ninguna'}")

    print("\n[DOSIS FERTILIZANTE] Variables detectadas:")
    print(f"  {dosis_detectada if dosis_detectada else '⚠️ Ninguna'}")

    return df

def resolver_oferta(df: pd.DataFrame, mode: str, custom_suffixes=None) -> dict:
    all_cols = [
        c for c in df.columns
        if c.endswith(("_kg_ha_min", "_kg_ha_avg", "_kg_ha_max"))
    ]

    cols_min = [c for c in all_cols if c.endswith("_kg_ha_min")]
    cols_avg = [c for c in all_cols if c.endswith("_kg_ha_avg")]
    cols_max = [c for c in all_cols if c.endswith("_kg_ha_max")]

    if mode == "all":
        selected = all_cols

    elif mode == "min":
        selected = cols_min

    elif mode == "avg":
        selected = cols_avg

    elif mode == "max":
        selected = cols_max

    elif mode == "custom":
        suffixes = {
            s.strip().lower().replace("_kg_ha_", "")
            for s in (custom_suffixes or [])
        }

        selected = []

        if "min" in suffixes:
            selected += cols_min

        if "avg" in suffixes:
            selected += cols_avg

        if "max" in suffixes:
            selected += cols_max

        selected = list(dict.fromkeys(selected))

    else:
        raise ValueError(f"oferta_mode='{mode}' no válido.")

    selected = [
        c for c in selected
        if c != "oferta_bases_kg_ha_avg"
    ]

    return {
        "selected": selected,
        "min": [c for c in selected if c.endswith("_kg_ha_min")],
        "avg": [c for c in selected if c.endswith("_kg_ha_avg")],
        "max": [c for c in selected if c.endswith("_kg_ha_max")],
    }

def validar_separacion_suelo_foliar(
        nutrientes_suelo: list,
        foliares: list,
):
    suelo_set = set(nutrientes_suelo)
    foliar_set = set(foliares)

    overlap = suelo_set.intersection(foliar_set)

    if overlap:
        raise ValueError(
            "ERROR CRÍTICO: se detectaron columnas simultáneamente clasificadas "
            f"como suelo y foliares: {sorted(overlap)}"
        )

    invalid_foliar = [
        c for c in foliares
        if not c.startswith("fol_") and not c.startswith("tipo_fol_")
    ]

    if invalid_foliar:
        raise ValueError(
            "ERROR CRÍTICO: variables foliares sin prefijo seguro `fol_`: "
            f"{invalid_foliar}"
        )

def preparar_features(df: pd.DataFrame):
    df = df.copy()

    categorical_columns = [
        "finca",
        "departamento",
        "material",
        "variedad",
        "textura_es",
        "tipo_de_analise_foliar",
        "fertilizantes",
        "produto_comercial",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalizar_texto)

    df = df.drop(columns=["textura_en"], errors="ignore")

    bma_originales = [
        c for c in df.columns
        if c.startswith("bma_")
    ]

    bma_ordinales = []

    for col in bma_originales:
        new_col = f"{col}_ord"

        df[new_col] = (
            df[col]
            .astype(str)
            .str.upper()
            .map(BMA_MAP)
        )

        bma_ordinales.append(new_col)

    df = df.drop(columns=bma_originales, errors="ignore")

    nutrientes_suelo = [
        c for c in [
            "ph",
            "cea",
            "ce",
            "mo",
            "p",
            "p_meh",
            "p_res",
            "p_rem",
            "p_total",
            "ca",
            "mg",
            "k",
            "na",
            "al",
            "cic",
            "cice",
            "acid_int",
            "sat_ca",
            "sat_mg",
            "sat_k",
            "sat_na",
            "sat_al",
            "sat_bases",
            "ca_mg",
            "mg_k",
            "ca_k",
            "ca_mg_k",
            "k_na",
            "psi",
            "ras",
            "fe_mn",
            "b",
            "cu",
            "fe",
            "mn",
            "zn",
            "s",
            "a",
            "l",
            "ar",
            "kgn_ton",
            "kgp_ton",
            "kgk_ton",
        ]
        if c in df.columns
    ]

    oferta_info = resolver_oferta(
        df,
        "avg",
        "avg",
    )

    oferta = oferta_info["selected"]

    dist_opt = [
        c for c in df.columns
        if c.startswith("dist_opt_")
    ]

    score_opt = [
        c for c in df.columns
        if c.startswith("score_")
    ]

    indices_suelo = [
        c for c in [
            "indice_acidez",
            "indice_bases",
            "indice_sodicidad",
            "indice_micros",
            "indice_fertilidad_quimica",
            "ifc",
            "bases_suma",
            "bases_ca_mg_k",
            "rel_al_bases",
            "rel_na_bases",
            "ca_pct_cic",
            "mg_pct_cic",
            "k_pct_cic",
            "na_pct_cic",
            "micro_balance_b_zn",
            "micro_balance_fe_zn",
            "micro_balance_cu_zn",
            "textura_fina",
            "rel_arena_arcilla",
        ]
        if c in df.columns
    ]

    foliares_numericos = [
        c for c in df.columns
        if c.startswith("fol_")
           and pd.api.types.is_numeric_dtype(df[c])
    ]

    foliares = list(dict.fromkeys(
        foliares_numericos
    ))

    fertilizacion_nutricional = [
        c for c in [
            "kg_ha_n",
            "kg_n_ha",
            "kg_ha_p",
            "kg_p_ha",
            "kg_ha_k",
            "kg_k_ha",
            "kg_ha_mg",
            "kg_mg_ha",
            "kg_ha_s",
            "kg_s_ha",
            "kg_ha_b",
            "kg_b_ha",
            "kg_ha_zn",
            "kg_zn_ha",
            "kg_palma",
            "dosagem_kg_planta",
        ]
        if c in df.columns
    ]

    fertilizacion_nutricional = list(dict.fromkeys(
        fertilizacion_nutricional
    ))

    climatologia = [
        c for c in [
            "pp_anual",
            "temp_promedio",
            "humedad_relativa_promedio",
            "brillo_solar_promedio",
        ]
        if c in df.columns
    ]

    anomalias_plagas = [
        c for c in [
            "fratura_copa",
            "amarelecimento_fatal",
            "anel_vermelho",
            "opsiphanes_invirae",
        ]
        if c in df.columns
    ]

    validar_separacion_suelo_foliar(
        nutrientes_suelo=nutrientes_suelo,
        foliares=foliares_numericos,
    )

    suelos_all = list(dict.fromkeys(
        nutrientes_suelo
        + oferta
        + dist_opt
        + score_opt
        + indices_suelo
        + bma_ordinales
    ))

    return df, suelos_all, fertilizacion_nutricional, foliares, climatologia

# ════════════════════════════════════════════════════
# 2. FUNCIONES DE GRÁFICOS
# ════════════════════════════════════════════════════

def plot_serie_climatologica_anual(
    df: pd.DataFrame,
    var: str,
    group_col: str = "Ano",
) -> go.Figure:

    if var not in df.columns:
        return px.scatter(title=f"Variable {var} no disponible.")

    if group_col not in df.columns:
        return px.scatter(title=f"No existe la columna temporal '{group_col}'.")

    d = df[[group_col, var]].copy()
    d[var] = pd.to_numeric(d[var], errors="coerce")
    d[group_col] = pd.to_numeric(d[group_col], errors="coerce")
    d = d.dropna(subset=[group_col, var])

    if d.empty:
        return px.scatter(title=f"Sin datos numéricos disponibles para {var}.")

    serie_anual = (
        d.groupby(group_col, as_index=False)[var]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                group_col: "Ano",
                "mean": "Promedio",
                "median": "Mediana",
                "std": "Desv_estandar",
                "count": "N_registros",
            }
        )
        .sort_values("Ano")
    )

    if serie_anual.empty:
        return px.scatter(title=f"Sin datos anuales para {var}.")

    global_mean = float(d[var].mean())

    fig = go.Figure()

    if serie_anual["Desv_estandar"].notna().any():
        upper = serie_anual["Promedio"] + serie_anual["Desv_estandar"].fillna(0)
        lower = serie_anual["Promedio"] - serie_anual["Desv_estandar"].fillna(0)

        fig.add_trace(go.Scatter(
            x=serie_anual["Ano"],
            y=upper,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=serie_anual["Ano"],
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.14)",
            hoverinfo="skip",
            name="± 1 DE",
        ))

    fig.add_trace(go.Scatter(
        x=serie_anual["Ano"],
        y=serie_anual["Promedio"],
        mode="lines+markers",
        name="Promedio anual",
        line=dict(color=COLORS["primary"], width=3),
        marker=dict(size=9, color=COLORS["primary"]),
        customdata=np.stack(
            [
                serie_anual["Mediana"].fillna(np.nan),
                serie_anual["Desv_estandar"].fillna(np.nan),
                serie_anual["N_registros"].fillna(0),
            ],
            axis=-1,
        ),
        hovertemplate=(
            "<b>Año %{x}</b><br>"
            f"Promedio {var}: %{{y:.2f}}<br>"
            "Mediana: %{customdata[0]:.2f}<br>"
            "DE: %{customdata[1]:.2f}<br>"
            "N registros: %{customdata[2]}<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=serie_anual["Ano"],
        y=serie_anual["Mediana"],
        mode="lines+markers",
        name="Mediana anual",
        line=dict(color=COLORS["warning"], width=2, dash="dot"),
        marker=dict(size=6, color=COLORS["warning"]),
        hovertemplate=(
            "<b>Año %{x}</b><br>"
            f"Mediana {var}: %{{y:.2f}}<extra></extra>"
        ),
    ))

    fig.add_hline(
        y=global_mean,
        line_dash="dash",
        line_color=COLORS["success"],
        line_width=2,
        annotation_text=f"Media global: {global_mean:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        height=470,
        margin=dict(t=60, l=0, r=0, b=0),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title="Año",
        tickmode="linear",
        dtick=1,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        title=var,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )

    return fig

def calcular_cuadrantes(df: pd.DataFrame, prod_col: str = "ton_ha",
                        group_cols=("lote","finca")) -> pd.DataFrame:
    gc = [c for c in group_cols if c in df.columns]
    if not gc or prod_col not in df.columns:
        return pd.DataFrame()

    agg = df.groupby(gc)[prod_col].agg(media="mean", de="std").reset_index()
    agg = agg.dropna(subset=["media","de"])
    if agg.empty:
        return agg

    med_g = agg["media"].mean()
    de_g = agg["de"].mean()

    def asignar_q(media, de, med_ref, de_ref):
        if de < de_ref and media < med_ref: return "Q2"
        elif de < de_ref and media >= med_ref: return "Q4"
        elif de >= de_ref and media < med_ref: return "Q1"
        else: return "Q3"

    agg["q_global"] = agg.apply(lambda r: asignar_q(r["media"], r["de"], med_g, de_g), axis=1)

    finca_col = "finca" if "finca" in gc else gc[0]
    agg["q_local"] = ""
    for finca, sub in agg.groupby(finca_col):
        med_f = sub["media"].mean()
        de_f = sub["de"].mean()
        for idx in sub.index:
            agg.at[idx, "q_local"] = asignar_q(agg.at[idx,"media"], agg.at[idx,"de"], med_f, de_f)

    return agg

def fig_scatter_cuadrantes(df_q: pd.DataFrame, q_col: str,
                            x_col: str, y_col: str,
                            x_label: str, y_label: str,
                            title: str, label_col: str = None) -> go.Figure:
    if df_q.empty:
        return go.Figure()
    fig = go.Figure()
    mapping = {"q1":"brown","q2":"orange","q3":"green","q4":"dodgerblue"}
    for q, color in mapping.items():
        sub = df_q[df_q[q_col] == q.upper()]
        if sub.empty:
            continue
        text_vals = sub[label_col].astype(str).tolist() if label_col and label_col in sub.columns else [""]*len(sub)
        fig.add_trace(go.Scatter(
            x=sub[x_col], y=sub[y_col], mode="markers",
            name=q.upper(), marker=dict(color=color, size=9, opacity=0.85, line=dict(color="white", width=0.8)),
            text=text_vals,
            hovertemplate=f"<b>%{{text}}</b><br>{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}<br>Cuadrante: {q.upper()}<extra></extra>"
        ))
    mx = df_q[x_col].mean(); my = df_q[y_col].mean()
    fig.add_vline(x=mx, line_dash="dash", line_color="black", line_width=0.8,
                  annotation_text=f"μ={mx:.2f}", annotation_position="top right", annotation_font_size=10)
    fig.add_hline(y=my, line_dash="dash", line_color="black", line_width=0.8,
                  annotation_text=f"σ={my:.2f}", annotation_position="bottom right", annotation_font_size=10)
    # Anotaciones
    xmn, xmx = df_q[x_col].min(), df_q[x_col].max()
    ymn, ymx = df_q[y_col].min(), df_q[y_col].max()
    annot = [
        ("Q1\nBaja prod.\nAlta var.", (xmn+mx)/2, (my+ymx)/2, "brown"),
        ("Q2\nBaja prod.\nBaja var.", (xmn+mx)/2, (ymn+my)/2, "orange"),
        ("Q3\nAlta prod.\nAlta var.", (mx+xmx)/2, (my+ymx)/2, "green"),
        ("Q4\nAlta prod.\nBaja var.", (mx+xmx)/2, (ymn+my)/2, "dodgerblue"),
    ]
    for txt, ax, ay, col in annot:
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False, font=dict(size=9, color=col), opacity=0.5)
    fig.update_layout(title=dict(text=title, font_size=13),
                      xaxis_title=x_label, yaxis_title=y_label, height=480, margin=dict(t=50,l=0,r=0,b=0),
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.12))
    return fig

def fig_boxplot(df, x_col, y_col, title="", global_mean=None) -> go.Figure:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    order = df.groupby(x_col)[y_col].median().sort_values().index.tolist()
    fig = go.Figure()
    for cat in order:
        vals = df[df[x_col]==cat][y_col].dropna().tolist()
        fig.add_trace(go.Box(y=vals, name=str(cat), marker_color=COLORS["primary"], showlegend=False, boxmean=False))
    if global_mean is not None:
        fig.add_hline(y=global_mean, line_dash="dash", line_color="red",
                      annotation_text=f"Media global: {global_mean:.2f}", annotation_position="top right")
    fig.update_layout(height=280, margin=dict(t=10,l=0,r=0,b=0), xaxis_tickangle=-45, showlegend=False, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",  plot_bgcolor = "rgba(0,0,0,0)")
    return fig

def fig_distplot(vals: pd.Series, label: str) -> go.Figure:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    fig = go.Figure()
    if len(vals) < 2:
        return fig
    fig.add_trace(go.Histogram(x=vals, histnorm="probability density",
                               marker_color=COLORS["primary"], opacity=0.55,
                               nbinsx=25, name="Distribución"))
    if vals.nunique() >= 3:
        try:
            kde_x = np.linspace(vals.min(), vals.max(), 300)
            kde_y = gaussian_kde(vals)(kde_x)
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines",
                                     line=dict(color=COLORS["primary"], width=2), name="KDE"))
        except Exception:
            pass
    fig.add_vline(x=float(vals.mean()), line_dash="dash", line_color=COLORS["danger"],
                  annotation_text=f"μ={vals.mean():.2f}", annotation_position="top right")
    fig.update_layout(height=280, margin=dict(t=0,l=0,r=0,b=0), xaxis_title=label, yaxis_title="Densidad", showlegend=False, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",  plot_bgcolor = "rgba(0,0,0,0)")

    return fig

def fig_radar_bma(df_finca: pd.DataFrame) -> go.Figure:
    bma_avail = [c for c in BMA_MAP if c in df_finca.columns]
    if not bma_avail:
        return go.Figure()
    pct = {}
    for c in bma_avail:
        col_val = df_finca[c].dropna()
        pct[c] = round((col_val == 1).sum() / len(col_val) * 100, 1) if len(col_val) else 0
    labels = [c.replace("bma_","").upper() for c in bma_avail]
    values = list(pct.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]], fill="toself",
        line=dict(color=COLORS["primary"], width=2), marker=dict(color=COLORS["primary"]), name="% lotes en BMA"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                      height=380, margin=dict(t=30,l=20,r=20,b=20), paper_bgcolor="rgba(0,0,0,0)",
                      title="% Lotes en Rango BMA (Best Management Area)")
    return fig

def fig_heatmap_ifc(df: pd.DataFrame) -> go.Figure:
    if "ifc" not in df.columns:
        return go.Figure()
    grp_cols = [c for c in ["departamento","finca"] if c in df.columns]
    if len(grp_cols) < 2:
        return go.Figure()
    pivot = df.groupby(grp_cols)["ifc"].mean().unstack(fill_value=np.nan)
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1, text_auto=".2f",
                    labels=dict(x="finca", y="departamento", color="IFC"))
    fig.update_layout(height=380, margin=dict(t=30,l=0,r=0,b=0), title="IFC por Departamento × Finca", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _find_soil_var(df_cols: list, target: str) -> str | None:
    cols = {c: c for c in df_cols}
    cols_low = {c.lower(): c for c in df_cols}
    target_low = target.lower()
    if target in cols:
        return target
    if target_low in cols_low:
        return cols_low[target_low]
    for c in df_cols:
        if target_low in c.lower():
            return c
    return None

def _resolve_soil_group(df: pd.DataFrame, group_cols: tuple) -> tuple:
    out = []
    for g in group_cols:
        real = _find_soil_var(df.columns, g)
        if real is None:
            return None
        out.append(real)
    return tuple(out)

def _coerce_soil_numeric(df: pd.DataFrame) -> pd.DataFrame:
    all_vars = {c for cols in DIMENSIONES_CORR.values() for c in cols}
    for v in all_vars:
        real = _find_soil_var(df.columns, v)
        if real and not pd.api.types.is_numeric_dtype(df[real]):
            df[real] = pd.to_numeric(
                df[real].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    return df

# ════════════════════════════════════════════════════
# 3. SECCIONES DEL DASHBOARD
# ════════════════════════════════════════════════════

def kpi_card(col, label, value, icon=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
    </div>""", unsafe_allow_html=True)

def seccion_kpis(data: pd.DataFrame):
    years    = sorted(data["ano"].dropna().unique())
    year_max = int(years[-1]) if years else "—"
    cols = st.columns(6)
    kpis = [
        ("Año de análisis",     str(year_max),                        "📅"),
        ("Fincas",              str(data["finca"].nunique()),          "🏡"),
        ("Lotes analizados",    str(data["lote"].nunique()),           "🌿"),
        ("Registros",           f"{len(data):,}",                      "📋"),
        ("TON/HA promedio",     f"{data['ton_ha'].mean():.2f}",        "📊"),
    ]
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

def _bloques_vars(df: pd.DataFrame):
    """Detecta variables por bloque usando los patrones del pipeline."""
    cols = set(df.columns)
    suelo = [
        c for c in cols
        if c in {
            "ph","mo","p","p_meh","p_res","p_total","ca","mg","k","na","al","cic","cice",
            "acid_int","sat_ca","sat_mg","sat_k","sat_na","sat_al","b","cu","fe","mn","zn",
            "s","a","l","ar","ifc","indice_fertilidad_quimica",
        }
    ]
    fert = [
        c for c in cols
        if c.startswith("kg_ha_") or c in {
            "kg_n_ha","kg_p_ha","kg_k_ha","kg_mg_ha","kg_s_ha","kg_b_ha","kg_zn_ha",
            "dosagem_kg_planta",
        }
    ]
    foliar = [c for c in cols if c.startswith("fol_")]
    clima = [c for c in cols if c in {
        "pp_anual","temp_promedio","humedad_relativa_promedio","brillo_solar_promedio",
    }]
    anom = [c for c in cols if c in {
        "fratura_copa","amarelecimento_fatal","anel_vermelho","opsiphanes_invirae",
    }]
    return {
        "Suelo": suelo,
        "Fertilización": fert,
        "Foliar": foliar,
        "Clima": clima,
        "Anomalías/Plagas": anom,
    }

def _corr_bloque(df: pd.DataFrame, var_list: list, min_obs: int = 15) -> pd.Series:
    """Correlación de cada variable del bloque con ton_ha."""
    if "ton_ha" not in df.columns:
        return pd.Series(dtype=float)
    out = {}
    for c in var_list:
        if c not in df.columns:
            continue
        tmp = df[[c, "ton_ha"]].dropna()
        if len(tmp) >= min_obs and tmp[c].nunique() > 1:
            out[c] = tmp[c].corr(tmp["ton_ha"])
    return pd.Series(out).dropna()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — resolución robusta de DIMENSIONES_CORR
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar_nombre_col(nombre) -> str:
    """
    Normaliza nombres para comparar columnas sin depender de:
    mayúsculas, minúsculas, tildes, espacios, guiones, slash o guion bajo.

    Ejemplos:
    'pH' / 'PH' / 'p_h'       -> 'ph'
    'Acid_int' / 'ACID INT'   -> 'acidint'
    'Ca_Mg__K' / 'ca-mg-k'    -> 'camgk'
    """
    texto = str(nombre).strip().lower()

    reemplazos = str.maketrans({
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    })
    texto = texto.translate(reemplazos)

    return re.sub(r"[^a-z0-9]+", "", texto)


def _resolver_columna_dim(df_columns, variable_dim):
    """
    Busca una variable de DIMENSIONES_CORR en las columnas reales del dataset.

    La comparación es exacta después de normalizar. No usa substring,
    porque variables cortas como P, K, S o Ca podrían confundirse con
    columnas como Prof, Sample, etc.
    """
    objetivo = _normalizar_nombre_col(variable_dim)

    for col in df_columns:
        if _normalizar_nombre_col(col) == objetivo:
            return col

    return None


def _resolver_columna(df_columns, candidatos):
    """
    Resuelve nombres conocidos por coincidencia exacta normalizada.
    """
    candidatos_norm = {_normalizar_nombre_col(c) for c in candidatos}

    for col in df_columns:
        if _normalizar_nombre_col(col) in candidatos_norm:
            return col

    return None


def _convertir_serie_numerica_suelo(serie: pd.Series) -> pd.Series:
    """
    Convierte columnas con formatos frecuentes:
    2,35       -> 2.35
    1.234,56   -> 1234.56
    1,234.56   -> 1234.56
    """
    s = serie.astype(str).str.strip()

    # Vacíos y representaciones frecuentes de nulos
    s = s.replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "NULL": np.nan,
        "null": np.nan,
        "N/A": np.nan,
        "n/a": np.nan,
        "-": np.nan,
    })

    tiene_coma = s.str.contains(",", regex=False, na=False)
    tiene_punto = s.str.contains(".", regex=False, na=False)

    # Formato latino: 1.234,56
    mascara_latina = tiene_coma & tiene_punto & (
        s.str.rfind(",") > s.str.rfind(".")
    )

    # Formato internacional: 1,234.56
    mascara_internacional = tiene_coma & tiene_punto & ~mascara_latina

    s_limpia = s.copy()

    # 1.234,56 -> 1234.56
    s_limpia.loc[mascara_latina] = (
        s_limpia.loc[mascara_latina]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    # 1,234.56 -> 1234.56
    s_limpia.loc[mascara_internacional] = (
        s_limpia.loc[mascara_internacional]
        .str.replace(",", "", regex=False)
    )

    # 2,35 -> 2.35
    mascara_solo_coma = tiene_coma & ~tiene_punto
    s_limpia.loc[mascara_solo_coma] = (
        s_limpia.loc[mascara_solo_coma]
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(s_limpia, errors="coerce")


def _forzar_numericas_dimensiones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte a numéricas todas las columnas reales encontradas
    dentro de DIMENSIONES_CORR.
    """
    df_out = df.copy()

    variables_dim = {
        variable
        for grupo in DIMENSIONES_CORR.values()
        for variable in grupo
    }

    for variable in variables_dim:
        col_real = _resolver_columna_dim(df_out.columns, variable)

        if col_real is not None:
            df_out[col_real] = _convertir_serie_numerica_suelo(
                df_out[col_real]
            )

    return df_out


def _grupos_dimensiones_reales(df: pd.DataFrame) -> dict:
    """
    Devuelve DIMENSIONES_CORR usando los nombres REALES de las columnas
    existentes y numéricas del dataframe.

    Ejemplo:
    DIMENSIONES_CORR: pH, CIC, Acid_int
    DataFrame:        PH, cic, ACID INT

    Resultado:
    {'Reacción...': ['PH', 'cic', 'ACID INT']}
    """
    grupos = {}

    for nombre_grupo, variables_dim in DIMENSIONES_CORR.items():
        variables_reales = []

        for variable in variables_dim:
            col_real = _resolver_columna_dim(df.columns, variable)

            if (
                col_real is not None
                and pd.api.types.is_numeric_dtype(df[col_real])
                and df[col_real].notna().any()
            ):
                variables_reales.append(col_real)

        # Eliminar duplicados preservando el orden declarado en DIMENSIONES_CORR
        variables_reales = list(dict.fromkeys(variables_reales))

        if variables_reales:
            grupos[nombre_grupo] = variables_reales

    return grupos


# ─────────────────────────────────────────────────────────────────────────────
# BAR + TABLE + PIE NUMÉRICOS PARA CORRELACIÓN CON TON/HA
# ─────────────────────────────────────────────────────────────────────────────

def make_bar_table_pie(df_corr: pd.DataFrame, value_label: str):
    """
    Construye tres visuales numéricos a partir de un DataFrame con:
    - Variable
    - Correlacion
    - N_pares

    No usa conteos categóricos. Todo se calcula con el coeficiente
    de correlación de Pearson y su magnitud absoluta.
    """
    X = df_corr.copy()

    if X.empty:
        return go.Figure(), go.Figure(), go.Figure()

    X["Correlacion"] = pd.to_numeric(X["Correlacion"], errors="coerce")
    X["N_pares"] = pd.to_numeric(X["N_pares"], errors="coerce")

    X = X.dropna(subset=["Correlacion"]).copy()

    if X.empty:
        return go.Figure(), go.Figure(), go.Figure()

    X["Magnitud"] = X["Correlacion"].abs()
    X = X.sort_values("Correlacion", ascending=True).reset_index(drop=True)

    colores = np.where(
        X["Correlacion"] >= 0,
        COLORS["success"],
        COLORS["danger"],
    )

    # ── 1. Barras: coeficiente de correlación firmado ──
    bar = go.Figure()

    bar.add_trace(
        go.Bar(
            x=X["Correlacion"],
            y=X["Variable"],
            orientation="h",
            marker_color=colores,
            text=X["Correlacion"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Correlación: %{x:.3f}<extra></extra>"
            ),
        )
    )

    bar.add_vline(
        x=0,
        line_color="#7A8899",
        line_width=1.2,
    )

    bar.update_layout(
        title="Correlación de variables con ton/ha",
        height=max(260, 32 * len(X) + 110),
        margin=dict(t=42, l=10, r=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            title=value_label,
            range=[-1, 1],
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
    )

    # ── 2. Tabla: correlación + pares válidos ──
    tabla = X.sort_values("Magnitud", ascending=False).copy()

    table = go.Figure(
        go.Table(
            header=dict(
                values=[
                    "Variable",
                    "Correlación",
                    "Magnitud |r|",
                    "Pares válidos",
                ],
                fill_color=COLORS["primary"],
                font=dict(color="white", size=11),
                align=["left", "center", "center", "center"],
            ),
            cells=dict(
                values=[
                    tabla["Variable"],
                    tabla["Correlacion"].map(lambda x: f"{x:.3f}"),
                    tabla["Magnitud"].map(lambda x: f"{x:.3f}"),
                    tabla["N_pares"].fillna(0).astype(int),
                ],
                fill_color=[
                    [
                        "#F7F9FC" if i % 2 == 0 else "white"
                        for i in range(len(tabla))
                    ]
                ],
                align=["left", "center", "center", "center"],
                font=dict(size=10),
            ),
            columnwidth=[160, 95, 95, 95],
        )
    )

    table.update_layout(
        title="Detalle numérico de correlaciones",
        height=max(260, 34 * len(tabla) + 110),
        margin=dict(t=42, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # ── 3. Dona: peso relativo según magnitud absoluta |r| ──
    top_pie = tabla.head(8).copy()
    total_magnitud = top_pie["Magnitud"].sum()

    if total_magnitud > 0:
        pie = go.Figure(
            go.Pie(
                labels=top_pie["Variable"],
                values=top_pie["Magnitud"],
                hole=0.58,
                textinfo="label+percent",
                textposition="outside",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "|r|: %{value:.3f}<br>"
                    "Participación: %{percent}<extra></extra>"
                ),
            )
        )

        pie.update_layout(
            title="Peso relativo de |correlación|",
            height=max(260, 32 * len(top_pie) + 110),
            margin=dict(t=42, l=10, r=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[
                dict(
                    text="Top<br>|r|",
                    x=0.5,
                    y=0.5,
                    font=dict(size=15, color="#1C2B3A"),
                    showarrow=False,
                )
            ],
        )
    else:
        pie = go.Figure()
        pie.add_annotation(
            text="No hay magnitudes de correlación<br>distintas de cero.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        pie.update_layout(
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
        )

    return bar, table, pie


# ─────────────────────────────────────────────────────────────────────────────
# TAB RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

def tab_resumen(df: pd.DataFrame):
    st.markdown(
        '<div class="section-title">🔗 Correlación con ton/ha</div>',
        unsafe_allow_html=True,
    )

    # Trabajar sobre copia y convertir correctamente las variables de DIMENSIONES_CORR
    df_work = _forzar_numericas_dimensiones(df)

    # Resolver ton/ha sin depender de mayúsculas o formato del nombre
    prod_col = _resolver_columna(
        df_work.columns,
        [
            "ton_ha",
            "ton/ha",
            "ton ha",
            "tonha",
            "rendimiento",
            "produccion",
            "producción",
        ],
    )

    if prod_col is not None:
        df_work[prod_col] = _convertir_serie_numerica_suelo(df_work[prod_col])

    # Recalcular; no reutilizar numeric_cols viejo de session_state
    numeric_cols = (
        df_work.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    if not numeric_cols:
        st.info("No se detectaron columnas numéricas para el análisis de correlación.")
        return

    # DIMENSIONES_CORR resuelto contra nombres reales del dataframe
    grupos_cols = _grupos_dimensiones_reales(df_work)
    grupos_validos = list(grupos_cols.keys())

    if grupos_validos:
        group_options = grupos_validos + ["Custom"]
        default_group = grupos_validos[0]
    else:
        group_options = ["Custom"]
        default_group = "Custom"

    if (
        "resumen_corr_group_select" not in st.session_state
        or st.session_state["resumen_corr_group_select"] not in group_options
    ):
        st.session_state["resumen_corr_group_select"] = default_group

    if "resumen_corr_cols_tab" not in st.session_state:
        if default_group != "Custom":
            st.session_state["resumen_corr_cols_tab"] = (
                grupos_cols[default_group].copy()
            )
        else:
            st.session_state["resumen_corr_cols_tab"] = (
                numeric_cols[:min(len(numeric_cols), 12)]
            )

    def _aplicar_grupo_resumen():
        grupo_sel = st.session_state.get(
            "resumen_corr_group_select",
            default_group,
        )

        if grupo_sel == "Custom":
            st.session_state["resumen_corr_cols_tab"] = (
                numeric_cols[:min(len(numeric_cols), 12)]
            )
        else:
            st.session_state["resumen_corr_cols_tab"] = (
                grupos_cols.get(grupo_sel, []).copy()
            )

    col_sel, col_info = st.columns([2, 1])

    with col_sel:
        st.selectbox(
            "Grupo de variables de suelo:",
            options=group_options,
            key="resumen_corr_group_select",
            on_change=_aplicar_grupo_resumen,
        )

    with col_info:
        grupo_actual = st.session_state.get(
            "resumen_corr_group_select",
            default_group,
        )

        if grupo_actual == "Custom":
            st.caption("Selección manual de variables.")
        else:
            st.caption(
                f"{len(grupos_cols.get(grupo_actual, []))} "
                "variables reconocidas."
            )

    seleccion_actual = [
        c
        for c in st.session_state.get("resumen_corr_cols_tab", [])
        if c in numeric_cols
    ]
    st.session_state["resumen_corr_cols_tab"] = seleccion_actual

    corr_cols = st.multiselect(
        "Columnas de suelo para el heatmap:",
        options=numeric_cols,
        key="resumen_corr_cols_tab",
    )

    heatmap_cols = corr_cols.copy()

    if prod_col is not None and prod_col not in heatmap_cols:
        heatmap_cols = [prod_col] + heatmap_cols

    heatmap_cols = list(dict.fromkeys(heatmap_cols))

    if len(heatmap_cols) < 2:
        st.info(
            "Selecciona al menos dos variables de suelo para construir "
            "la matriz de correlación."
        )
        return

    sub_df = (
        df_work[heatmap_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )

    if sub_df.shape[0] < 3:
        st.info(
            f"No hay suficientes filas válidas ({sub_df.shape[0]}) "
            "para calcular correlaciones."
        )
        return

    corr = sub_df.corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        labels=dict(
            x="Variables",
            y="Variables",
            color="Correlación",
        ),
    )

    fig_corr.update_traces(textfont=dict(size=11))
    fig_corr.update_xaxes(tickangle=45, side="bottom")

    fig_corr.update_layout(
        height=max(480, 43 * len(corr.columns) + 130),
        margin=dict(t=20, b=90, l=20, r=20),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title="Correlación",
            thickness=15,
        ),
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True,
        key=sanitize_key("resumen_corr_heatmap"),
    )

    st.download_button(
        "⬇️ Descargar matriz de correlación (CSV)",
        corr.round(3).to_csv().encode("utf-8"),
        file_name="correlacion_resumen.csv",
        mime="text/csv",
        key=sanitize_key("dl_corr_resumen"),
    )

    if prod_col is None or prod_col not in corr.columns:
        st.info(
            "No se detectó una columna de producción para calcular "
            "correlaciones individuales con ton/ha."
        )
        return

    variables_suelo = [
        c
        for c in corr.columns
        if c != prod_col
    ]

    if not variables_suelo:
        st.info(
            "Selecciona al menos una variable de suelo adicional a ton/ha."
        )
        return

    filas_corr = []

    for variable in variables_suelo:
        pares_validos = (
            sub_df[[prod_col, variable]]
            .dropna()
            .shape[0]
        )

        coef = corr.loc[variable, prod_col]

        if pd.notna(coef):
            filas_corr.append(
                {
                    "Variable": variable,
                    "Correlacion": float(coef),
                    "N_pares": int(pares_validos),
                }
            )

    df_corr_ton = pd.DataFrame(filas_corr)

    if df_corr_ton.empty:
        st.info(
            "No se encontraron pares numéricos válidos para correlacionar "
            "las variables seleccionadas con ton/ha."
        )
        return

    st.markdown(
        '<div class="section-title">📈 Intensidad de relación con ton/ha</div>',
        unsafe_allow_html=True,
    )

    fig_bar, fig_table, fig_pie = make_bar_table_pie(
        df_corr_ton,
        value_label="Coeficiente de correlación (r)",
    )

    col_bar, col_table, col_pie = st.columns([1.25, 1.2, 1])

    with col_bar:
        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            key=sanitize_key("resumen_corr_bar"),
        )

    with col_table:
        st.plotly_chart(
            fig_table,
            use_container_width=True,
            key=sanitize_key("resumen_corr_table"),
        )

    with col_pie:
        st.plotly_chart(
            fig_pie,
            use_container_width=True,
            key=sanitize_key("resumen_corr_pie"),
        )

    st.download_button(
        "⬇️ Descargar correlaciones con ton/ha (CSV)",
        df_corr_ton.sort_values(
            "Correlacion",
            key=lambda s: s.abs(),
            ascending=False,
        ).to_csv(index=False).encode("utf-8"),
        file_name="correlaciones_con_ton_ha.csv",
        mime="text/csv",
        key=sanitize_key("dl_corr_ton_ha"),
    )

def tab_produccion(df: pd.DataFrame):
    st.markdown('<div class="section-title">Análisis Exploratorio de Distribución y Variabilidad de la Productividad</div>',
                    unsafe_allow_html=True)
    if "ton_ha" not in df.columns:
        st.info("Columna ton/ha no disponible.")
        return
    c_sel1, c_sel2, c_sel3 = st.columns(3)
    agrupar = c_sel1.selectbox("Agrupar por:", ["finca","departamento","lote","material"], key="prod_group")
    c1, c2 = st.columns(2)
    with c1:
        fig = fig_boxplot(df, agrupar, "ton_ha", f"{"ton_ha"} por {agrupar}", global_mean=df["ton_ha"].mean() if "ton_ha" in df.columns else None)
        st.plotly_chart(fig, use_container_width=True, key=sanitize_key(f"prod_box_{"ton_ha"}_{agrupar}"))
    with c2:
        fig2 = fig_distplot(df["ton_ha"], "ton_ha")
        st.plotly_chart(fig2, use_container_width=True, key=sanitize_key(f"prod_dist_{"ton_ha"}"))
    st.markdown('<div class="section-title"> Mapas de árbol o jerárquico</div>',
                unsafe_allow_html=True)
    if "departamento" in df.columns and "finca" in df.columns:
        df_tm = df.dropna(subset=["ton_ha"])
        if not df_tm.empty:
            fig_tree = px.treemap(
                df_tm,
                path=[px.Constant("Total"), "departamento", "finca", "lote"],
                hover_data=["ton_ha"],
                custom_data=["material", "area"],
                values=df_tm["ton_ha"].abs() + 0.01,
                color=df_tm["ton_ha"],
                color_continuous_scale="Spectral",
                color_continuous_midpoint=np.average(
                    df_tm["ton_ha"].replace([np.inf, -np.inf], np.nan).dropna()
                ),
            )
            fig_tree.data[0].textinfo = "label+text+value"
            fig_tree.data[0].texttemplate = (
                "MATERIAL: %{customdata[0]}<br>"
                "AREA: %{customdata[1]:.1f} ha<br>"
            )
            fig_tree.update_traces(hovertemplate="%{label}<br>%{value:.2f}")
            fig_tree.update_layout(height=320, template="plotly_white", margin=dict(t=10, l=0, r=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)")
            st.plotly_chart(fig_tree, use_container_width=True, key="prod_treemap")
    st.markdown("---")

def tab_suelos(df: pd.DataFrame, SUELO_COLS):
    st.markdown('<div class="section-title">Análisis Exploratorio de Distribución y Variabilidad de los Nutrientes del suelo</div>',
                    unsafe_allow_html=True)
    avail_suelo = [c for c in SUELO_COLS if c in df.columns]
    if not avail_suelo:
        st.info("No se encontraron variables de suelo.")
        return
    c_sel1, c_sel2, c_sel3 = st.columns(3)
    var = c_sel1.selectbox("Variable de suelo:", avail_suelo, key="suelo_var")
    agrupar = c_sel2.selectbox("Agrupar por:", ["finca","departamento","lote","material"], key="suelo_group")
    c1, c2 = st.columns(2)
    with c1:
        fig = fig_boxplot(df, agrupar, var, f"{var} por {agrupar}", global_mean=df[var].mean() if var in df.columns else None)
        st.plotly_chart(fig, use_container_width=True, key=sanitize_key(f"suelo_box_{var}_{agrupar}"))
    with c2:
        fig2 = fig_distplot(df[var], var)
        st.plotly_chart(fig2, use_container_width=True, key=sanitize_key(f"suelo_dist_{var}"))
    st.markdown("---")

def tab_fertilizacion(df: pd.DataFrame, FERT_COLS):
    st.markdown(
        '<div class="section-title">Análisis Exploratorio de Distribución y Variabilidad de las variables de Fertilización</div>',
        unsafe_allow_html=True,
    )

    avail_fert = [c for c in FERT_COLS if c in df.columns]
    if not avail_fert:
        st.info("No se encontraron columnas de fertilización.")
        return

    # ── Etiquetas legibles, robustas frente a kg_ha_, kg_x_ha, _kg_ha, etc. ──
    def _etiqueta_nutriente(col: str) -> str:
        etiqueta = re.sub(r"^kg_ha_", "", col)
        etiqueta = re.sub(r"_kg_ha$", "", etiqueta)
        etiqueta = re.sub(r"^kg_", "", etiqueta)
        etiqueta = re.sub(r"_ha$", "", etiqueta)
        etiqueta = etiqueta.replace("dosagem_kg_planta", "DOSIS_KG_PLANTA")
        return etiqueta.upper()

    nut_map = {c: _etiqueta_nutriente(c) for c in avail_fert}

    # ── Columnas de agrupación conectadas al dataframe filtrado del sidebar ──
    grupos_candidatos = ["finca", "departamento", "lote", "material"]
    grupos_disponibles = [c for c in grupos_candidatos if c in df.columns]

    if not grupos_disponibles:
        st.info(
            "No se encontraron columnas de agrupación (finca, departamento, "
            "lote o material) en los datos filtrados."
        )
        return

    c_sel1, c_sel2, c_sel3 = st.columns(3)

    nut_sel = c_sel1.selectbox(
        "Nutriente:",
        avail_fert,
        format_func=lambda x: nut_map[x],
        key="fert_nut",
    )

    agrupar = c_sel2.selectbox(
        "Agrupar por:",
        grupos_disponibles,
        key="fert_group",
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = fig_boxplot(
            df,
            agrupar,
            nut_sel,
            f"{nut_map[nut_sel]} por {agrupar}",
            global_mean=df[nut_sel].mean() if nut_sel in df.columns else None,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=sanitize_key(f"fert_box_{nut_sel}_{agrupar}"),
        )

    with c2:
        fig2 = fig_distplot(df[nut_sel], nut_map[nut_sel])
        st.plotly_chart(
            fig2,
            use_container_width=True,
            key=sanitize_key(f"fert_dist_{nut_sel}"),
        )

    st.markdown(
        f'<div class="section-title">Aplicación media de nutrientes por {agrupar.capitalize()} (kg/ha)</div>',
        unsafe_allow_html=True,
    )

    # ── Usa la MISMA agrupación seleccionada arriba, no "finca" fijo ──
    agg_fert = df.groupby(agrupar)[avail_fert].mean().reset_index()

    fig_stack = px.bar(
        agg_fert,
        x=agrupar,
        y=avail_fert,
        barmode="group",
        labels={"value": "kg/ha", "variable": "Nutriente", agrupar: agrupar.capitalize()},
    )
    fig_stack.for_each_trace(
        lambda t: t.update(name=nut_map.get(t.name, t.name))
    )
    fig_stack.update_layout(
        height=280,
        template="plotly_white",
        margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(
        fig_stack,
        use_container_width=True,
        key=sanitize_key(f"fert_stacked_{agrupar}"),
    )

    st.markdown("---")

def tab_foliares(df: pd.DataFrame, FOLIAR_COLS):
    st.markdown('<div class="section-title">Análisis Exploratorio de Distribución y Variabilidad de las variables Foliares</div>',
                    unsafe_allow_html=True)
    avail_fol = [c for c in FOLIAR_COLS if c in df.columns]
    if not avail_fol:
        st.info("No hay datos foliares en el dataset.")
        return
    c_sel1, c_sel2, c_sel3 = st.columns(3)
    var_fol = c_sel1.selectbox("Variable foliar:", avail_fol, format_func=lambda x: x.replace("_f","").upper(), key="fol_var")
    agrupar = c_sel2.selectbox("Agrupar por:", ["finca","departamento","lote","material"], key="fol_group")
    c1, c2 = st.columns(2)
    with c1:
        fig = fig_boxplot(df, agrupar, var_fol, "", global_mean=df[var_fol].mean() if var_fol in df.columns else None)
        st.plotly_chart(fig, use_container_width=True, key=sanitize_key(f"fol_box_{var_fol}"))
    with c2:
        fig2 = fig_distplot(df[var_fol], var_fol)
        st.plotly_chart(fig2, use_container_width=True, key=sanitize_key(f"fol_dist_{var_fol}"))
    st.markdown("---")

def tab_clima(df: pd.DataFrame, CLIMA_COLS):
    st.markdown(
        '<div class="section-title">'
        'Análisis Exploratorio de Distribución y Variabilidad de las variables de Climatología'
        '</div>',
        unsafe_allow_html=True,
    )

    if df is None or df.empty:
        st.info("No hay datos disponibles para el análisis climatológico.")
        return

    avail_clima = [c for c in CLIMA_COLS if c in df.columns]

    if not avail_clima:
        st.info("No se encontraron variables de clima.")
        return

    year_col = None
    for candidate in ["Ano", "ano", "Año", "año", "year"]:
        if candidate in df.columns:
            year_col = candidate
            break

    if year_col is None:
        st.warning(
            "No se encontró una columna temporal anual. "
            "Se requiere una columna como `Ano`, `Año`, `ano` o `year`."
        )
        return

    df_clima = df.copy()

    if year_col != "Ano":
        df_clima = df_clima.rename(columns={year_col: "Ano"})

    df_clima["Ano"] = pd.to_numeric(df_clima["Ano"], errors="coerce")
    df_clima = df_clima.dropna(subset=["Ano"])
    df_clima["Ano"] = df_clima["Ano"].astype(int)

    if df_clima.empty:
        st.warning("No hay años válidos para construir la serie temporal.")
        return

    grupos_disponibles = [
        c for c in ["finca", "departamento", "lote", "material"]
        if c in df_clima.columns
    ]

    c_sel1, c_sel2, c_sel3 = st.columns(3)

    with c_sel1:
        var = st.selectbox(
            "Variable de clima:",
            avail_clima,
            key="clima_variable_seleccionada",
        )

    with c_sel2:
        if grupos_disponibles:
            agrupar = st.selectbox(
                "Agrupar por:",
                grupos_disponibles,
                key="clima_grupo_seleccionado",
            )
        else:
            agrupar = None
            st.caption("Sin columnas de agrupación disponibles.")

    with c_sel3:
        st.empty()

    c1, c2 = st.columns(2)

    with c1:
        if agrupar is not None:
            fig_box = fig_boxplot(
                df_clima,
                agrupar,
                var,
                f"{var} por {agrupar}",
                global_mean=df_clima[var].mean() if var in df_clima.columns else None,
            )
            st.plotly_chart(
                fig_box,
                use_container_width=True,
                key=sanitize_key(f"clima_box_{var}_{agrupar}"),
            )
        else:
            st.info("No hay una columna disponible para agrupar el boxplot.")

    with c2:
        fig_dist = fig_distplot(df_clima[var], var)
        st.plotly_chart(
            fig_dist,
            use_container_width=True,
            key=sanitize_key(f"clima_dist_{var}"),
        )

    st.markdown(
        '<div class="section-title">Serie climatológica anual</div>',
        unsafe_allow_html=True,
    )

    fig_serie = plot_serie_climatologica_anual(
        df=df_clima,
        var=var,
        group_col="Ano",
    )

    st.plotly_chart(
        fig_serie,
        use_container_width=True,
        key=sanitize_key(f"clima_serie_anual_{var}"),
    )

    st.markdown("---")

def tab_cuadrantes(df: pd.DataFrame):
    st.markdown(
        '<div class="section-title">Análisis de Cuadrantes</div>',
        unsafe_allow_html=True,
    )

    prod_col = next(
        (
            c
            for c in df.columns
            if c.lower() in {"ton_ha", "ton/ha", "ton_ha_pred", "rendimiento", "produccion"}
        ),
        None,
    )
    if prod_col is None:
        st.info("Se requiere una columna de producción (ton_ha, TON/HA, etc.).")
        return

    c1, c2 = st.columns(2)
    modo_q = c1.radio(
        "Escala de referencia:",
        ["Global", "Por Finca"],
        horizontal=True,
        key="q_modo",
    )
    group_q = c2.selectbox(
        "Nivel de agrupación:",
        ["Lote + Finca", "Lote", "Finca"],
        key="q_group",
    )

    group_map = {
        "Lote + Finca": ("lote", "finca"),
        "Lote": ("lote",),
        "Finca": ("finca",),
    }
    gc_key = group_map[group_q]

    df_q = calcular_cuadrantes(df, prod_col=prod_col, group_cols=gc_key)
    if df_q.empty:
        st.warning("No hay suficientes datos para calcular cuadrantes.")
        return

    q_col = "q_global" if modo_q == "Global" else "q_local"
    label_col = _find_soil_var(df_q.columns, gc_key[0]) if gc_key else None

    st.markdown(
        '<div class="section-title">Mapa de Cuadrantes (Productividad vs Variabilidad)</div>',
        unsafe_allow_html=True,
    )

    fig_q = fig_scatter_cuadrantes(
        df_q,
        q_col=q_col,
        x_col="media",
        y_col="de",
        x_label="Productividad media (ton/ha)",
        y_label="Variabilidad (DE)",
        title=f"Cuadrantes — Nivel: {group_q} · Referencia: {modo_q}",
        label_col=label_col,
    )
    st.plotly_chart(
        fig_q,
        use_container_width=True,
        key=sanitize_key(f"cuadrantes_scatter_{modo_q}_{group_q}"),
    )

    df = _coerce_soil_numeric(df.copy())

    grupos_reales = {}
    for gname, cols in DIMENSIONES_CORR.items():
        avail = []
        for c in cols:
            real = _find_soil_var(df.columns, c)
            if real and pd.api.types.is_numeric_dtype(df[real]):
                avail.append(real)
        if avail:
            grupos_reales[gname] = avail

    if not grupos_reales:
        st.info(
            "No se detectaron variables de suelo numéricas. "
            "Solo se mostrará el resumen de cuadrantes."
        )
        resumen_solo = df_q.groupby(q_col).agg(
            N_lotes=("media", "count"),
            Prod_media=("media", "mean"),
            DE_media=("de", "mean"),
        ).round(2).reset_index()
        st.dataframe(
            resumen_solo,
            use_container_width=True,
            key="cuadrantes_resumen_solo",
        )
        return

    st.markdown(
        '<div class="section-title">Perfil de Suelo por Cuadrante</div>',
        unsafe_allow_html=True,
    )

    col_ctrl, col_table = st.columns([1, 2])

    with col_ctrl:
        st.markdown(
            '<div class="kpi-label">Controles del perfil</div>',
            unsafe_allow_html=True,
        )

        block_options = list(grupos_reales.keys()) + ["Custom"]
        selected_block = st.selectbox(
            "Bloque de variables de suelo:",
            block_options,
            index=0,
            key="q_soil_block",
        )

        if selected_block == "Custom":
            all_avail = sorted({c for cols in grupos_reales.values() for c in cols})
            soil_vars = st.multiselect(
                "Variables a incluir:",
                options=all_avail,
                default=all_avail[: min(len(all_avail), 8)],
                key="q_soil_vars_custom",
            )
        else:
            soil_vars = grupos_reales[selected_block]
            st.caption(f"Variables: {', '.join(soil_vars)}")

        filtro_q = st.multiselect(
            "Filtrar cuadrante(s) en el heatmap:",
            ["Q1", "Q2", "Q3", "Q4"],
            default=["Q1", "Q2", "Q3", "Q4"],
            key="q_filter",
        )

    with col_table:
        st.markdown(
            '<div class="section-title">Resumen por Cuadrante</div>',
            unsafe_allow_html=True,
        )

        resumen = df_q.groupby(q_col).agg(
            N_lotes=("media", "count"),
            Prod_media=("media", "mean"),
            DE_media=("de", "mean"),
        ).round(2).reset_index()

        fig_resumen = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(resumen.columns),
                        fill_color=COLORS["primary"],
                        font=dict(color="white", size=12),
                        align="left",
                    ),
                    cells=dict(
                        values=[resumen[c] for c in resumen.columns],
                        fill_color=[
                            ["#f7f9fc" if i % 2 == 0 else "white"]
                            for i in range(len(resumen))
                        ],
                        align="left",
                        font=dict(size=11),
                    ),
                )
            ]
        )
        fig_resumen.update_layout(
            height=160 + 40 * len(resumen),
            margin=dict(t=0, l=0, r=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(
            fig_resumen,
            use_container_width=True,
            key=sanitize_key(f"cuadrantes_resumen_table_{modo_q}_{group_q}"),
        )

    st.markdown(
        '<div class="section-title">Mapa de Calor — Perfil de Suelo por Cuadrante</div>',
        unsafe_allow_html=True,
    )

    if not soil_vars or label_col is None:
        st.info(
            "No se pueden cruzar cuadrantes con variables de suelo. "
            "Verifica la agrupación y las variables seleccionadas."
        )
        return

    df_q_heat = df_q[df_q[q_col].isin(filtro_q)].copy()
    if df_q_heat.empty:
        st.info("No hay cuadrantes seleccionados para el perfil de suelo.")
        return

    merge_on = _find_soil_var(df.columns, gc_key[0])
    df_merged = df.merge(
        df_q_heat[[label_col, q_col]],
        left_on=merge_on,
        right_on=label_col,
        how="left",
    )
    perfil = df_merged.groupby(q_col)[soil_vars].mean().round(3)

    if perfil.empty or perfil.shape[1] == 0:
        st.info(
            "No hay datos suficientes para calcular el perfil de suelo "
            "con los cuadrantes y variables seleccionados."
        )
        return

    fig_perf = px.imshow(
        perfil,
        color_continuous_scale="RdYlGn",
        text_auto=".2f",
        aspect="auto",
        labels=dict(x="Variable", y="Cuadrante", color="Media"),
    )
    fig_perf.update_traces(textfont=dict(size=10))
    fig_perf.update_xaxes(tickangle=45, side="bottom")
    fig_perf.update_layout(
        height=max(320, 60 * len(perfil.index) + 120),
        margin=dict(t=20, b=80, l=20, r=20),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="Media", thickness=15),
    )
    st.plotly_chart(
        fig_perf,
        use_container_width=True,
        key=sanitize_key(
            f"cuadrantes_perfil_{modo_q}_{group_q}_{selected_block}"
        ),
    )

def tab_modelo(df: pd.DataFrame, SUELO_COLS, FERT_COLS, FOLIAR_COLS):
    st.markdown('<div class="section-title">Modelo Predictivo — Random Forest (ton/ha)</div>', unsafe_allow_html=True)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
    except Exception:
        st.error("scikit-learn no está instalado. Ejecuta pip install scikit-learn")
        return
    if "ton_ha" not in df.columns:
        st.info("ton_ha requerido.")
        return
    feature_pool = (SUELO_COLS + FERT_COLS + FOLIAR_COLS + ["edad","densidad","area"])
    features_avail = [c for c in feature_pool if c in df.columns]
    c_opt1, c_opt2 = st.columns(2)
    n_est = c_opt1.slider("Nº de árboles:", 50, 500, 150, 50, key="rf_nest")
    min_feat = c_opt2.slider("% mínimo de datos por feature:", 10, 80, 30, key="rf_minfeat")
    df_model = df[features_avail + ["ton_ha"]].dropna(subset=["ton_ha"])
    df_model = df_model.select_dtypes(include=[np.number])
    good_feats = [c for c in df_model.columns if c != "ton_ha" and df_model[c].notna().sum() >= len(df_model) * (min_feat/100)]
    df_model = df_model[good_feats + ["ton_ha"]].dropna()
    if len(df_model) < 50 or len(good_feats) < 3:
        st.warning(f"Pocos datos ({len(df_model)} filas, {len(good_feats)} features).")
        return
    st.info(f"Entrenando con {len(df_model):,} registros y {len(good_feats)} variables.")
    with st.spinner("Entrenando Random Forest..."):
        X = df_model[good_feats].values
        y = df_model["ton_ha"].values
        rf = RandomForestRegressor(n_estimators=n_est, max_depth=8, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        y_pred = rf.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
    ck1, ck2, ck3 = st.columns(3)
    kpi_card(ck1, "R² (entrenamiento)", f"{r2:.3f}", "📈")
    kpi_card(ck2, "RMSE (ton_ha)", f"{rmse:.3f}", "📉")
    kpi_card(ck3, "Features usadas", str(len(good_feats)), "🔢")
    st.markdown("---")
    st.markdown('<div class="section-title">⚡ Reglas condicionales — Top variables</div>', unsafe_allow_html=True)
    imp_desc = pd.DataFrame({"variable": good_feats, "importancia": rf.feature_importances_}).sort_values("importancia",
                                                                                                      ascending=False)
    top5 = imp_desc["variable"].tolist()[:5]
    if top5:
        umbrales_cols = st.columns(len(top5))
        for i, feat in enumerate(top5):
            vals = df_model[feat].dropna()
            if len(vals) == 0: continue
            p25, p50, p75 = vals.quantile([.25, .5, .75])
            low_mask = df_model[feat] <= p25
            high_mask = df_model[feat] >= p75
            mu_low = df_model.loc[low_mask, "ton_ha"].mean()
            mu_high = df_model.loc[high_mask, "ton_ha"].mean()
            delta = mu_high - mu_low
            color = COLORS["success"] if delta > 0 else COLORS["danger"]
            umbrales_cols[i].markdown(f"""
                <div class="kpi-card" style="border-left-color:{color};">
                    <div class="kpi-label">{feat}</div>
                    <div style="font-size:.8rem;margin-top:.3rem;">
                        P25 ≤ {p25:.2f}: media <b>{mu_low:.2f} ton/ha</b><br>
                        P75 ≥ {p75:.2f}: media <b>{mu_high:.2f} ton/ha</b><br>
                        <span style="color:{color};font-weight:700;">Δ = {delta:+.2f} ton/ha</span>
                    </div>
                </div>""", unsafe_allow_html=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        imp = pd.DataFrame({"variable": good_feats, "importancia": rf.feature_importances_}).sort_values("importancia", ascending=True).tail(20)
        fig_imp = px.bar(imp, x="importancia", y="variable", orientation="h", color="importancia", color_continuous_scale="Greens")
        st.plotly_chart(fig_imp, use_container_width=True, key="rf_importance")
    with c2:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=y, y=y_pred, mode="markers", marker=dict(color=COLORS["primary"], opacity=0.5, size=5), name="Lotes"))
        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
        fig_pred.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="red", dash="dash"), name="Ideal"))
        fig_pred.update_layout(xaxis_title="Real (ton/ha)", yaxis_title="Predicho (ton/ha)", height=520)
        st.plotly_chart(fig_pred, use_container_width=True, key="rf_pred_vs_real")
    st.markdown("---")

# ════════════════════════════════════════════════════
# 4. SIDEBAR
# ════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 FarmPrecision")
        st.markdown("---")
        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel o CSV",
            type=["xlsx", "xls", "csv"],
            help="El archivo debe contener columnas de finca, lote, año, siembra, material, área y ton/ha.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")

        st.markdown("""
        | **Variable** | **Nombre interno** | **Variantes aceptadas** |
        |:-------------|:-------------------|:-------------------------|
        | **Finca** | `finca` | finca, farm, hacienda |
        | **Lote** | `lote` | lote, lot, parcela |
        | **Año** | `ano` | año, ano, year |
        | **Año de siembra** | `ano_siembra` | siembra, ano_siembra, año_siembra |
        | **Material** | `material` | material, variedad, genotipo |
        | **Área (ha)** | `area` | area_ha, area, ha |
        | **Producción (ton/ha)** | `ton_ha` | ton/ha, ton_ha, rff/ha, rendimiento |
        """)

        st.caption(
            "El sistema normaliza automáticamente mayúsculas, minúsculas, "
            "acentos, espacios y caracteres especiales."
        )

    return uploaded

# ════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════

def main():
    uploaded = render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>Dashboard Principal — FarmPrecision</h1>
        <p>Producción · Suelos · Fertilización · Foliares · Climatología · Cuadrantes · Modelo</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel o CSV para comenzar</h3>
            <p>Usa el panel lateral para cargar el archivo.</p>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls · .csv</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")
        c1, c2, c3,c4 = st.columns(4)
        c1.info("**Aplicación anual** de nutrientes (N, P, K, Ca, Mg, S, B, Zn) por año y lote")
        c2.info("**Perfil nutricional** por edad del cultivo y ranking de lotes por nutriente")
        c3.info("**Correlaciones** entre macronutrientes, micronutrientes e intensidad total")
        c4.info("**Estadísticas** descriptivas, distribuciones, correlaciones y tendencias")
        return

    with st.spinner("⏳ Procesando datos..."):
        try:
            data = cargar_dataset_consolidado(uploaded.read(), file_name=uploaded.name)
            data, SUELO_COLS, FERT_COLS, FOLIAR_COLS, CLIMA_COLS = preparar_features(data)
        except ValueError as e:
            st.error(f"❌ Error al cargar el archivo:\n\n{e}")
            return
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
            return

    if data.empty:
        st.warning("El archivo no contiene datos válidos después del filtrado.")
        return

    with st.sidebar:
        fincas_all = sorted(data["finca"].dropna().unique().tolist()) if "finca" in data.columns else []
        departamentos_all = sorted(data["departamento"].dropna().unique().tolist()) if "departamento" in data.columns else []
        anos_all = sorted(pd.to_numeric(data["ano"], errors="coerce").dropna().unique().astype(int).tolist()) if "ano" in data.columns else []
        mats_all = sorted(data["material"].dropna().unique().tolist()) if "material" in data.columns else []
        departamento_sel = st.multiselect("Departamento:", departamentos_all, default=departamentos_all, key="f_departamento")
        finca_sel = st.multiselect("Finca:", fincas_all, default=fincas_all, key="f_finca")
        año_sel = st.multiselect("Año:", anos_all, default=anos_all, key="f_ano")
        mat_sel = st.multiselect("Material:", mats_all, default=mats_all, key="f_mat")
        lote_q = st.text_input("Buscar Lote:", "", key="f_lote")

    df = data.copy()
    if departamento_sel and "departamento" in df.columns: df = df[df["departamento"].isin(departamento_sel)]
    if finca_sel and "finca" in df.columns: df = df[df["finca"].isin(finca_sel)]
    if año_sel and "ano" in df.columns: df = df[df["ano"].astype(int).isin(año_sel)]
    if mat_sel and "material" in df.columns: df = df[df["material"].isin(mat_sel)]
    if lote_q.strip() and "lote" in df.columns:
        df = df[df["lote"].astype(str).str.contains(lote_q.strip(), case=False, na=False)]

    st.sidebar.write(f"Registros filtrados: **{len(df):,}**")

    if df.empty:
        st.warning("Sin datos con los filtros actuales.")
        return

    seccion_kpis(df)

    st.markdown("---")

    tab_names = ["📌 Resumen","📊 Producción","🗺️ Suelos","🧪 Fertilización","🔬 Foliares", "📈 Climatología","🔲 Cuadrantes","📋 Modelo RF"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_resumen(df)
    with tabs[1]:
        tab_produccion(df)
    with tabs[2]:
        tab_suelos(df, SUELO_COLS)
    with tabs[3]:
        tab_fertilizacion(df, FERT_COLS)
    with tabs[4]:
        tab_foliares(df, FOLIAR_COLS)
    with tabs[5]:
        tab_clima(df, CLIMA_COLS)
    with tabs[6]:
        tab_cuadrantes(df)
    with tabs[7]:
        tab_modelo(df,SUELO_COLS, FERT_COLS, FOLIAR_COLS)

if __name__ == "__main__":
    main()