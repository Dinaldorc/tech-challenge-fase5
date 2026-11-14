import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import normalize_column_name, quality_report  # noqa: E402
from src.features import build_next_year_dataset, make_preprocessor, temporal_split  # noqa: E402
from src.predict import InputValidationError, RiscoAcompanhamentoModel, validate_record  # noqa: E402
from src.train import quality_gate  # noqa: E402

FEATURES = ["inde", "ieg", "fase"]


def _valid():
    return {"inde": 7.2, "ieg": 8.0, "fase": 3}


def test_registro_valido():
    assert validate_record(_valid(), FEATURES) == []


def test_nulos_sao_aceitos():
    assert validate_record({"inde": None, "ieg": 8.0, "fase": 3}, FEATURES) == []


def test_fora_da_faixa():
    errs = validate_record({**_valid(), "inde": 15.0}, FEATURES)
    assert any("fora da faixa" in e for e in errs)


def test_tipo_invalido():
    errs = validate_record({**_valid(), "ieg": "alto"}, FEATURES)
    assert any("não numérico" in e for e in errs)


def test_campo_inesperado():
    errs = validate_record({**_valid(), "nome": "Fulano"}, FEATURES)
    assert any("não esperados" in e for e in errs)


def test_todos_nulos():
    errs = validate_record({f: None for f in FEATURES}, FEATURES)
    assert "nenhuma feature informada" in errs


def _model():
    X = pd.DataFrame({"inde": [5, 6, 7, 8], "ieg": [5, 6, 7, 8], "fase": [1, 2, 3, 4]})
    y = [0, 1, 0, 1]
    pipe = Pipeline([("prep", make_preprocessor(FEATURES)), ("clf", DummyClassifier(strategy="prior"))]).fit(X, y)
    return RiscoAcompanhamentoModel(pipe, FEATURES, threshold=0.5)


def test_modelo_retorna_erro_por_linha_sem_quebrar():
    df = pd.DataFrame([_valid(), {**_valid(), "inde": 99.0}])
    out = _model().predict(None, df)
    assert list(out["status"]) == ["ok", "erro"]
    assert not math.isnan(out.loc[0, "probabilidade"])
    assert math.isnan(out.loc[1, "probabilidade"])
    assert out["aviso"].str.contains("Não é diagnóstico").all()


def test_entrada_vazia():
    with pytest.raises(InputValidationError):
        _model().predict(None, pd.DataFrame(columns=FEATURES))


def test_normalize_column_name():
    assert normalize_column_name("INDE 2023") == "inde"
    assert normalize_column_name("Instituição de ensino") == "instituicao_de_ensino"
    assert normalize_column_name("Defas") == "defas"


def test_alvo_vem_do_ano_seguinte_sem_vazamento():
    silver = pd.DataFrame(
        {
            config.ID_COL: ["a", "a", "b", "b"],
            config.YEAR_COL: [2022, 2023, 2022, 2023],
            "inde": [6.0, 7.0, 8.0, 5.0],
            "defasagem": [0, -1, 0, 0],
        }
    )
    ds = build_next_year_dataset(silver, "defasagem_next_year")
    assert set(ds[config.YEAR_COL]) == {2022}
    # features são do ano t (2022), alvo é a defasagem de 2023
    row_a = ds[ds[config.ID_COL] == "a"].iloc[0]
    assert row_a["inde"] == 6.0 and row_a["defasagem"] == 0 and row_a[config.TARGET_COL] == 1
    train, test = temporal_split(pd.concat([ds, ds.assign(**{config.YEAR_COL: 2023})]))
    assert train[config.YEAR_COL].max() < test[config.YEAR_COL].min()


def test_quality_report_detecta_fora_da_faixa():
    rep = quality_report(pd.DataFrame({"inde": [5.0, 11.0, np.nan]})).set_index("coluna")
    assert rep.loc["inde", "fora_da_faixa"] == 1
    assert rep.loc["inde", "pct_nulos"] == pytest.approx(33.33, 0.01)


def test_quality_gate():
    base = {"pr_auc": 0.40, "recall": 0.7, "precision": 0.3, "brier": 0.2}
    good = {"pr_auc": 0.50, "recall": 0.7, "precision": 0.4, "brier": 0.18}
    bad = {**good, "pr_auc": 0.41}
    assert quality_gate(good, base)[0]
    assert not quality_gate(bad, base)[0]
