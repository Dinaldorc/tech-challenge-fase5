"""Sinais de monitoramento: drift de features (PSI) e resumo do log de inferência."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_ALERT = 0.2  # > 0.2: mudança relevante de distribuição
PSI_WARN = 0.1


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = reference.dropna().to_numpy()
    cur = current.dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        value = psi(reference[f], current[f])
        status = "alerta" if value > PSI_ALERT else "atencao" if value > PSI_WARN else "ok"
        rows.append(
            {
                "feature": f,
                "psi": round(value, 4),
                "status": status,
                "pct_nulos_ref": round(reference[f].isna().mean() * 100, 2),
                "pct_nulos_atual": round(current[f].isna().mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)
