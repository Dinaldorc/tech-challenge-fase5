"""Validação de entrada e wrapper pyfunc servido no Model Serving.

O wrapper garante que, tanto no endpoint quanto no fallback local, a entrada é
validada e a saída é interpretável (probabilidade + faixa + aviso de uso).
"""
from __future__ import annotations

import math

import mlflow.pyfunc
import pandas as pd

from src import config

AVISO = (
    "Sinal de apoio à investigação pedagógica. Não é diagnóstico nem decisão "
    "sobre o estudante; deve ser avaliado pela equipe da Passos Mágicos."
)


class InputValidationError(ValueError):
    pass


def validate_record(record: dict, features: list[str]) -> list[str]:
    """Retorna lista de erros (vazia = registro válido). Nulos são permitidos (imputados)."""
    errors = []
    unknown = set(record) - set(features)
    if unknown:
        errors.append(f"campos não esperados: {sorted(unknown)}")
    if all(record.get(f) is None for f in features):
        errors.append("nenhuma feature informada")
    for f in features:
        v = record.get(f)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"{f}: valor não numérico ({v!r})")
            continue
        lo, hi = config.FEATURE_RANGES.get(f, (-math.inf, math.inf))
        if not lo <= v <= hi:
            errors.append(f"{f}: {v} fora da faixa [{lo}, {hi}]")
    return errors


def validate_frame(df: pd.DataFrame, features: list[str]) -> pd.Series:
    """Série com a lista de erros de cada linha."""
    if df.empty:
        raise InputValidationError("entrada vazia")
    records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return pd.Series([validate_record(r, features) for r in records], index=df.index)


def risk_band(p: float) -> str:
    if p >= 0.7:
        return "alta"
    if p >= 0.4:
        return "media"
    return "baixa"


class RiscoAcompanhamentoModel(mlflow.pyfunc.PythonModel):
    """Envolve o Pipeline sklearn com validação, limiar e saída explicável."""

    def __init__(self, pipeline=None, features: list[str] | None = None, threshold: float = 0.5):
        self.pipeline = pipeline
        self.features = features or []
        self.threshold = threshold

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        errors = validate_frame(model_input, self.features)
        out = pd.DataFrame(index=model_input.index)
        out["status"] = errors.map(lambda e: "erro" if e else "ok")
        out["erros"] = errors.map(lambda e: "; ".join(e))
        out["probabilidade"] = float("nan")
        out["sinal_apoio"] = None
        out["faixa"] = None

        valid = errors.map(len) == 0
        if valid.any():
            X = model_input.loc[valid, self.features].apply(pd.to_numeric, errors="coerce")
            proba = self.pipeline.predict_proba(X)[:, 1]
            out.loc[valid, "probabilidade"] = proba.round(4)
            out.loc[valid, "sinal_apoio"] = ["sim" if p >= self.threshold else "nao" for p in proba]
            out.loc[valid, "faixa"] = [risk_band(p) for p in proba]
        out["limiar"] = self.threshold
        out["aviso"] = AVISO
        return out.reset_index(drop=True)
