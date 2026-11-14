"""Ingestão, padronização e verificações de qualidade dos dados.

As funções trabalham com pandas para poderem ser testadas localmente. Nos
notebooks, o resultado é gravado em tabelas Delta (bronze/silver).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from src import config

# Nome bruto normalizado -> nome canônico. Ajustar após inspecionar a base real.
COLUMN_ALIASES = {
    "ra": "ra",
    "nome": "nome",
    "nome_anonimizado": "nome",
    "fase": "fase",
    "turma": "turma",
    "idade": "idade",
    "idade_aluno": "idade",
    "genero": "genero",
    "ano_ingresso": "ano_ingresso",
    "anos_pm": "anos_pm",
    "instituicao_de_ensino": "instituicao_ensino",
    "instituicao_ensino_aluno": "instituicao_ensino",
    "pedra": "pedra",
    "inde": "inde",
    "iaa": "iaa",
    "ieg": "ieg",
    "ips": "ips",
    "ipp": "ipp",
    "ida": "ida",
    "ipv": "ipv",
    "ian": "ian",
    "atingiu_pv": "atingiu_pv",
    "ponto_virada": "atingiu_pv",
    "fase_ideal": "fase_ideal",
    "defas": "defasagem",
    "defasagem": "defasagem",
    "indicado": "indicado",
    "rec_psicologia": "rec_psicologia",
}

_YEAR_SUFFIX = re.compile(r"[\s_]*(20)?2[0-9]$")


def normalize_column_name(name: str) -> str:
    """'INDE 2023' -> 'inde'; 'Instituição de ensino' -> 'instituicao_de_ensino'."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.strip().lower()
    s = _YEAR_SUFFIX.sub("", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def hash_id(value) -> str | None:
    """Pseudonimiza o identificador do estudante (não reversível sem a base original)."""
    if pd.isna(value):
        return None
    return hashlib.sha256(str(value).strip().encode()).hexdigest()[:16]


def to_numeric_br(series: pd.Series) -> pd.Series:
    """Converte números com vírgula decimal ('7,5') e textos inválidos para float."""
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(series, errors="coerce")


def standardize_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Converte a planilha de um ano para o esquema canônico (formato longo)."""
    renamed = {}
    for col in df.columns:
        norm = normalize_column_name(col)
        if norm in COLUMN_ALIASES and COLUMN_ALIASES[norm] not in renamed.values():
            renamed[col] = COLUMN_ALIASES[norm]
    out = df[list(renamed)].rename(columns=renamed).copy()
    out[config.YEAR_COL] = year

    for col in config.NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = to_numeric_br(out[col])

    if "fase" in out.columns:  # "ALFA" é a fase 0 em algumas versões da base
        out["fase"] = to_numeric_br(
            out["fase"].astype(str).str.upper().str.replace("ALFA", "0").str.extract(r"(\d+)")[0]
        )

    if "atingiu_pv" in out.columns:
        out["atingiu_pv"] = (
            out["atingiu_pv"].astype(str).str.strip().str.lower().map({"sim": 1, "nao": 0, "não": 0})
        )

    if "ra" in out.columns:
        out[config.ID_COL] = out["ra"].map(hash_id)
    return out


def drop_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identificadores diretos antes de gravar a camada silver."""
    return df.drop(columns=[c for c in ("nome", "ra") if c in df.columns])


def load_raw_excel(path: str) -> dict[str, pd.DataFrame]:
    """Lê todas as abas do Excel (uma aba por ano, ex.: PEDE2022, PEDE2023...)."""
    return pd.read_excel(path, sheet_name=None)


def year_from_sheet(sheet_name: str) -> int | None:
    m = re.search(r"(20\d{2})", sheet_name)
    return int(m.group(1)) if m else None


def build_silver(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in sheets.items():
        year = year_from_sheet(name)
        if year is None:
            continue
        frames.append(standardize_year(df, year))
    silver = pd.concat(frames, ignore_index=True)
    return drop_direct_identifiers(silver)


# ---------------------------------------------------------------------------
# Qualidade de dados (Requisito 2)
# ---------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por coluna: tipo, % nulos, distintos, fora da faixa plausível."""
    rows = []
    for col in df.columns:
        s = df[col]
        out_of_range = np.nan
        if col in config.FEATURE_RANGES and pd.api.types.is_numeric_dtype(s):
            lo, hi = config.FEATURE_RANGES[col]
            out_of_range = int(((s < lo) | (s > hi)).sum())
        rows.append(
            {
                "coluna": col,
                "dtype": str(s.dtype),
                "n": int(len(s)),
                "pct_nulos": round(float(s.isna().mean()) * 100, 2),
                "n_distintos": int(s.nunique(dropna=True)),
                "fora_da_faixa": out_of_range,
            }
        )
    return pd.DataFrame(rows)


def duplicate_check(df: pd.DataFrame) -> dict:
    keys = [config.ID_COL, config.YEAR_COL]
    keys = [k for k in keys if k in df.columns]
    return {
        "linhas_duplicadas_exatas": int(df.duplicated().sum()),
        "chave_duplicada_estudante_ano": int(df.duplicated(subset=keys).sum()) if keys else None,
    }


def assert_quality(df: pd.DataFrame, max_null_pct: float = 30.0) -> list[str]:
    """Retorna lista de problemas bloqueantes (vazia = ok)."""
    problems = []
    rep = quality_report(df)
    for _, r in rep.iterrows():
        if r["coluna"] in config.NUMERIC_FEATURES and r["pct_nulos"] > max_null_pct:
            problems.append(f"{r['coluna']}: {r['pct_nulos']}% nulos (> {max_null_pct}%)")
        if pd.notna(r["fora_da_faixa"]) and r["fora_da_faixa"] > 0:
            problems.append(f"{r['coluna']}: {int(r['fora_da_faixa'])} valores fora da faixa")
    dups = duplicate_check(df)
    if dups.get("chave_duplicada_estudante_ano"):
        problems.append(f"{dups['chave_duplicada_estudante_ano']} chaves estudante-ano duplicadas")
    return problems
