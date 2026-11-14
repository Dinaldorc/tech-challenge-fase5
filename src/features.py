"""Construção do alvo, das features e da divisão temporal.

Regra anti-vazamento: features vêm SEMPRE do ano t; o alvo vem do ano t+1.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def _target_defasagem(nxt: pd.DataFrame) -> pd.Series:
    # Defasagem negativa = estudante abaixo da fase ideal para a idade/escolaridade
    return (nxt["defasagem"] < 0).astype("Int64").where(nxt["defasagem"].notna())


def _target_not_pv(nxt: pd.DataFrame) -> pd.Series:
    return (1 - nxt["atingiu_pv"]).astype("Int64").where(nxt["atingiu_pv"].notna())


def _target_inde_drop(nxt: pd.DataFrame, cur_inde: pd.Series) -> pd.Series:
    return (nxt["inde"] < cur_inde).astype("Int64").where(nxt["inde"].notna() & cur_inde.notna())


TARGET_STRATEGIES = ("defasagem_next_year", "not_pv_next_year", "inde_drop_next_year")


def build_next_year_dataset(silver: pd.DataFrame, strategy: str = config.TARGET_STRATEGY) -> pd.DataFrame:
    """Junta (estudante, t) com (estudante, t+1) e cria o alvo observado em t+1.

    Estudantes sem registro em t+1 ficam fora (não há rótulo observável).
    Essa perda deve ser quantificada no notebook 01 (possível viés de evasão).
    """
    if strategy not in TARGET_STRATEGIES:
        raise ValueError(f"strategy deve ser uma de {TARGET_STRATEGIES}")

    cur = silver.copy()
    nxt = silver.copy()
    nxt[config.YEAR_COL] = nxt[config.YEAR_COL] - 1  # alinha t+1 com t
    merged = cur.merge(nxt, on=[config.ID_COL, config.YEAR_COL], suffixes=("", "_next"), how="inner")

    nxt_view = merged[[c for c in merged.columns if c.endswith("_next")]]
    nxt_view.columns = [c[: -len("_next")] for c in nxt_view.columns]

    if strategy == "defasagem_next_year":
        merged[config.TARGET_COL] = _target_defasagem(nxt_view)
    elif strategy == "not_pv_next_year":
        merged[config.TARGET_COL] = _target_not_pv(nxt_view)
    else:
        merged[config.TARGET_COL] = _target_inde_drop(nxt_view, merged["inde"])

    keep = [config.ID_COL, config.YEAR_COL] + feature_columns(merged) + [config.TARGET_COL]
    out = merged[keep].dropna(subset=[config.TARGET_COL]).reset_index(drop=True)
    out[config.TARGET_COL] = out[config.TARGET_COL].astype(int)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in config.NUMERIC_FEATURES if c in df.columns and c not in config.EXCLUDED_FROM_FEATURES]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Imputação e escala ajustadas apenas no treino (dentro do Pipeline)."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric, features)], remainder="drop")


def temporal_split(df: pd.DataFrame, test_year: int | None = None):
    """Treino = pares com origem antes de test_year; teste = origem em test_year.

    Por padrão, o último ano de origem disponível vira teste (simula uso real:
    treinar com o passado e prever o próximo ciclo).
    """
    years = sorted(df[config.YEAR_COL].unique())
    if len(years) < 2:
        raise ValueError("São necessários ao menos 2 anos de origem para a divisão temporal.")
    test_year = test_year or years[-1]
    train = df[df[config.YEAR_COL] < test_year].reset_index(drop=True)
    test = df[df[config.YEAR_COL] == test_year].reset_index(drop=True)
    return train, test
