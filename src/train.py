"""Treino, avaliação, escolha de limiar e quality gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src import config
from src.features import make_preprocessor


def build_models(features: list[str]) -> dict[str, Pipeline]:
    """Baselines + candidatos. Todos compartilham o mesmo pré-processamento."""
    rs = config.RANDOM_STATE
    return {
        # Baseline 1: prevê a prevalência (PR-AUC ~= taxa de positivos)
        "baseline_prior": Pipeline(
            [("prep", make_preprocessor(features)), ("clf", DummyClassifier(strategy="prior"))]
        ),
        # Baseline 2: regra simples — regressão logística só com o INDE do ano corrente
        "baseline_inde": Pipeline(
            [
                ("prep", make_preprocessor(["inde"])),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        ),
        "cand_logreg": Pipeline(
            [
                ("prep", make_preprocessor(features)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)),
            ]
        ),
        "cand_rf": Pipeline(
            [
                ("prep", make_preprocessor(features)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=rs
                    ),
                ),
            ]
        ),
        "cand_hgb": Pipeline(
            [
                ("prep", make_preprocessor(features)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.05, max_iter=300, class_weight="balanced", random_state=rs
                    ),
                ),
            ]
        ),
    }


def oof_predictions(model: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
    """Probabilidades out-of-fold com folds agrupados por estudante (sem vazamento entre anos)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in cv.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def choose_threshold(y_true, proba, target_recall: float = config.TARGET_RECALL_FOR_THRESHOLD) -> float:
    """Maior limiar que ainda garante o recall-alvo (prioriza não perder quem precisa de apoio)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(thresholds[ok[-1]])


def evaluate(y_true, proba, threshold: float) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        "pr_auc": float(average_precision_score(y_true, proba)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_both else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "prevalence": float(np.mean(y_true)),
    }


def quality_gate(candidate: dict, best_baseline: dict, criteria: dict = config.QUALITY_GATE) -> tuple[bool, list[dict]]:
    """Aplica os critérios definidos previamente em config.QUALITY_GATE."""
    checks = [
        {
            "criterio": "ganho PR-AUC sobre melhor baseline",
            "valor": candidate["pr_auc"] - best_baseline["pr_auc"],
            "limite": criteria["min_gain_over_baseline_pr_auc"],
            "ok": candidate["pr_auc"] - best_baseline["pr_auc"] >= criteria["min_gain_over_baseline_pr_auc"],
        },
        {"criterio": "recall mínimo", "valor": candidate["recall"], "limite": criteria["min_recall"],
         "ok": candidate["recall"] >= criteria["min_recall"]},
        {"criterio": "precisão mínima", "valor": candidate["precision"], "limite": criteria["min_precision"],
         "ok": candidate["precision"] >= criteria["min_precision"]},
        {"criterio": "Brier máximo", "valor": candidate["brier"], "limite": criteria["max_brier"],
         "ok": candidate["brier"] <= criteria["max_brier"]},
    ]
    passed = all(c["ok"] and not np.isnan(c["valor"]) for c in checks)
    return passed, checks


def segment_report(df: pd.DataFrame, y_true, proba, threshold: float, segment_col: str = "fase") -> pd.DataFrame:
    """Métricas por segmento para evitar que a média esconda erros relevantes."""
    rows = []
    tmp = df[[segment_col]].copy()
    tmp["y"] = np.asarray(y_true)
    tmp["p"] = np.asarray(proba)
    for seg, g in tmp.groupby(segment_col):
        if len(g) < 10:
            continue
        m = evaluate(g["y"].values, g["p"].values, threshold)
        rows.append({segment_col: seg, "n": len(g), **m})
    return pd.DataFrame(rows)
