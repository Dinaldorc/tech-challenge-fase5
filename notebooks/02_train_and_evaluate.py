# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Treino e avaliação com MLflow
# MAGIC
# MAGIC * Divisão **temporal**: treino = pares com origem nos anos anteriores; teste = último ano de origem.
# MAGIC * Seleção do candidato e do limiar **somente no treino** (validação cruzada agrupada por estudante).
# MAGIC * O teste é usado uma única vez, para o quality gate.
# MAGIC * Métrica principal: **PR-AUC** (classe positiva minoritária). Complementares: recall, precisão, F1,
# MAGIC   ROC-AUC e Brier (calibração).

# COMMAND ----------

import json
import os
import subprocess
import sys

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src import config
from src.features import feature_columns, temporal_split
from src.predict import RiscoAcompanhamentoModel
from src.train import build_models, choose_threshold, evaluate, oof_predictions, quality_gate, segment_report

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(config.EXPERIMENT_NAME)

# COMMAND ----------

gold = spark.table(config.TABLE_FEATURES).toPandas()
features = feature_columns(gold)
train, test = temporal_split(gold)
X_tr, y_tr, g_tr = train[features], train[config.TARGET_COL], train[config.ID_COL]
X_te, y_te = test[features], test[config.TARGET_COL]
print("features:", features)
print(f"treino={len(train)} (anos {sorted(train[config.YEAR_COL].unique())}, prev={y_tr.mean():.3f}) | "
      f"teste={len(test)} (ano {test[config.YEAR_COL].unique()}, prev={y_te.mean():.3f})")

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()
except Exception:
    git_commit = "indisponivel"

# COMMAND ----------

# MAGIC %md ## Validação cruzada no treino (baselines + candidatos)

# COMMAND ----------

results = {}
with mlflow.start_run(run_name="comparacao_modelos") as parent:
    mlflow.set_tags({"git_commit": git_commit, "target_strategy": config.TARGET_STRATEGY})
    mlflow.log_dict(config.QUALITY_GATE, "quality_gate_criteria.json")
    for name, model in build_models(features).items():
        with mlflow.start_run(run_name=name, nested=True) as run:
            oof = oof_predictions(model, X_tr, y_tr, g_tr)
            thr = choose_threshold(y_tr, oof)
            cv_metrics = evaluate(y_tr, oof, thr)
            mlflow.log_params({"model": name, "n_features": len(features), **{
                k: v for k, v in model.named_steps["clf"].get_params().items() if isinstance(v, (int, float, str))
            }})
            mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
            results[name] = {"run_id": run.info.run_id, "cv": cv_metrics, "threshold": thr}

cv_table = pd.DataFrame({k: v["cv"] for k, v in results.items()}).T.sort_values("pr_auc", ascending=False)
display(cv_table)

# COMMAND ----------

# MAGIC %md ## Escolha do candidato (pela CV do treino) e avaliação única no teste

# COMMAND ----------

baselines = [n for n in results if n.startswith("baseline")]
candidates = [n for n in results if n.startswith("cand")]
best_cand = max(candidates, key=lambda n: results[n]["cv"]["pr_auc"])
print("Candidato escolhido na CV:", best_cand)

models = build_models(features)
test_metrics = {}
fitted = {}
for name in baselines + [best_cand]:
    m = models[name].fit(X_tr, y_tr)
    fitted[name] = m
    proba = m.predict_proba(X_te)[:, 1]
    test_metrics[name] = evaluate(y_te, proba, results[name]["threshold"])

display(pd.DataFrame(test_metrics).T)
best_base = max(baselines, key=lambda n: test_metrics[n]["pr_auc"])
passed, checks = quality_gate(test_metrics[best_cand], test_metrics[best_base])
display(pd.DataFrame(checks))
print("QUALITY GATE:", "APROVADO" if passed else "REPROVADO")

# COMMAND ----------

# MAGIC %md ## Análise de erros por fase

# COMMAND ----------

cand_proba = fitted[best_cand].predict_proba(X_te)[:, 1]
seg = segment_report(test, y_te, cand_proba, results[best_cand]["threshold"], "fase")
display(seg)

# COMMAND ----------

# MAGIC %md ## Log do modelo final (pyfunc com validação de entrada) no run do candidato

# COMMAND ----------

wrapper = RiscoAcompanhamentoModel(fitted[best_cand], features, results[best_cand]["threshold"])
example = X_te.head(5)
signature = infer_signature(example, wrapper.predict(None, example))

with mlflow.start_run(run_id=results[best_cand]["run_id"]):
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics[best_cand].items()})
    mlflow.log_metrics({f"test_baseline_{k}": v for k, v in test_metrics[best_base].items()})
    mlflow.log_dict({"passed": passed, "best_baseline": best_base, "checks": checks}, "quality_gate_result.json")
    mlflow.log_table(seg, "segment_report_fase.json")
    mlflow.set_tags({"quality_gate": "passed" if passed else "failed", "git_commit": git_commit})
    info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapper,
        signature=signature,
        input_example=example,
        code_paths=[os.path.join(repo_root, "src")],
        pip_requirements=["mlflow", "scikit-learn", "pandas", "numpy"],
    )

handoff = {"run_id": results[best_cand]["run_id"], "model_uri": info.model_uri, "passed": passed,
           "candidate": best_cand, "features": features}
print(json.dumps(handoff, indent=2))
# O notebook 03 localiza este run pela tag `quality_gate` no experimento.
