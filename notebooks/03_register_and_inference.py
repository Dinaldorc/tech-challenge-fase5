# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 — Registro, deploy, inferência e monitoramento
# MAGIC
# MAGIC 1. Localiza o run mais recente do notebook 02 e lê o resultado do quality gate.
# MAGIC 2. Registra o modelo no Unity Catalog e só atribui o alias `champion` se o gate passou.
# MAGIC 3. Cria/atualiza o endpoint de Model Serving (se disponível no workspace).
# MAGIC 4. Inferência com entrada válida e inválida — via endpoint ou, se indisponível, via `pyfunc` (fallback documentado).
# MAGIC 5. Registra as previsões em `inference_log` e calcula drift (PSI).

# COMMAND ----------

import json
import os
import sys
from datetime import datetime, timezone

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src import config
from src.features import temporal_split
from src.monitoring import drift_report

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md ## 1. Run de origem e quality gate

# COMMAND ----------

exp = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.quality_gate != ''",
                          order_by=["start_time DESC"], max_results=1)
run_id = runs.iloc[0]["run_id"]
gate = mlflow.artifacts.load_dict(f"runs:/{run_id}/quality_gate_result.json")
print("run_id:", run_id)
display(pd.DataFrame(gate["checks"]))
print("Gate:", "APROVADO" if gate["passed"] else "REPROVADO")

# COMMAND ----------

# MAGIC %md ## 2. Registro no Unity Catalog

# COMMAND ----------

mv = mlflow.register_model(f"runs:/{run_id}/model", config.MODEL_NAME)
client.set_model_version_tag(config.MODEL_NAME, mv.version, "quality_gate", "passed" if gate["passed"] else "failed")
client.set_model_version_tag(config.MODEL_NAME, mv.version, "source_run_id", run_id)

if gate["passed"]:
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, mv.version)
    print(f"{config.MODEL_NAME} v{mv.version} promovido a @{config.CHAMPION_ALIAS}")
else:
    print(f"v{mv.version} registrada mas NÃO promovida (gate reprovado). @champion permanece onde estava.")

champion = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
print("Versão champion:", champion.version)

# COMMAND ----------

# MAGIC %md ## 3. Deploy — Model Serving

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
served = ServedEntityInput(entity_name=config.MODEL_NAME, entity_version=champion.version,
                           workload_size="Small", scale_to_zero_enabled=True)
SERVING_OK = False
try:
    existing = [e.name for e in w.serving_endpoints.list()]
    if config.SERVING_ENDPOINT_NAME in existing:
        w.serving_endpoints.update_config_and_wait(name=config.SERVING_ENDPOINT_NAME, served_entities=[served])
    else:
        w.serving_endpoints.create_and_wait(name=config.SERVING_ENDPOINT_NAME,
                                            config=EndpointCoreConfigInput(served_entities=[served]))
    SERVING_OK = True
    print("Endpoint pronto:", config.SERVING_ENDPOINT_NAME)
except Exception as e:
    print("Model Serving indisponível neste workspace — usando fallback pyfunc. Motivo:", e)

# COMMAND ----------

# MAGIC %md ## 4. Inferência (entrada válida + entrada inválida)

# COMMAND ----------

gold = spark.table(config.TABLE_FEATURES).toPandas()
features = mlflow.models.get_model_info(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}") \
    .signature.inputs.input_names()
_, test = temporal_split(gold)

valid_rows = test[features].head(3).astype(float)
invalid_rows = pd.DataFrame([{**valid_rows.iloc[0].to_dict(), "inde": 42.0}])  # fora da faixa [0, 10]
payload = pd.concat([valid_rows, invalid_rows], ignore_index=True)
display(payload)


def score(df: pd.DataFrame) -> pd.DataFrame:
    if SERVING_OK:
        resp = w.serving_endpoints.query(name=config.SERVING_ENDPOINT_NAME,
                                         dataframe_records=json.loads(df.to_json(orient="records")))
        return pd.DataFrame(resp.predictions)
    model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
    return model.predict(df)


result = score(payload)
display(pd.concat([payload, result], axis=1))

# COMMAND ----------

# MAGIC %md ## 5. Log de inferência e drift

# COMMAND ----------

log = pd.concat([payload, result], axis=1)
log["model_version"] = str(champion.version)
log["canal"] = "endpoint" if SERVING_OK else "pyfunc_fallback"
log["ts"] = datetime.now(timezone.utc).isoformat()
spark.createDataFrame(log.astype({"sinal_apoio": "string", "faixa": "string"})) \
    .write.mode("append").option("mergeSchema", "true").saveAsTable(config.TABLE_INFERENCE_LOG)

train, test = temporal_split(gold)
display(drift_report(train, test, features))
