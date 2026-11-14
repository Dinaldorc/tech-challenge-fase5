# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Qualidade de dados e features
# MAGIC
# MAGIC **Fluxo:** arquivo bruto no Volume → `bronze_pede` (bruto, todas as abas) → `silver_pede` (padronizado,
# MAGIC sem identificadores diretos) → `gold_features` (pares ano *t* → *t+1* com alvo).
# MAGIC
# MAGIC **Pré-requisito:** enviar a base da Passos Mágicos para o Volume
# MAGIC `/Volumes/workspace/passos_magicos/raw/` (Catalog → workspace → passos_magicos → raw → Upload).

# COMMAND ----------

# MAGIC %pip install -q openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import sys

# Garante que `src` (raiz do repositório na Git folder) seja importável
repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd

from src import config
from src.data import assert_quality, build_silver, duplicate_check, load_raw_excel, quality_report
from src.features import build_next_year_dataset

# COMMAND ----------

# MAGIC %md ## 1. Catálogo, schema e volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.CATALOG}.{config.SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {config.CATALOG}.{config.SCHEMA}.{config.VOLUME}")
display(dbutils.fs.ls(config.VOLUME_PATH))

# COMMAND ----------

# MAGIC %md ## 2. Bronze — dado bruto, uma linha por registro de cada aba

# COMMAND ----------

raw_path = f"{config.VOLUME_PATH}/{config.RAW_FILE_NAME}"
sheets = load_raw_excel(raw_path)
for name, df in sheets.items():
    print(name, df.shape)

bronze = pd.concat(
    [df.astype(str).assign(_sheet=name) for name, df in sheets.items()], ignore_index=True
)
spark.createDataFrame(bronze).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_BRONZE)

# COMMAND ----------

# MAGIC %md ## 3. Silver — esquema canônico, tipos corrigidos, RA pseudonimizado, nome removido

# COMMAND ----------

silver = build_silver(sheets)
print(silver.shape)
display(silver.head(10))

# COMMAND ----------

# MAGIC %md ## 4. Verificações de qualidade (Requisito 2)

# COMMAND ----------

rep = quality_report(silver)
display(rep)
print(duplicate_check(silver))
problems = assert_quality(silver)
print("Problemas bloqueantes:" if problems else "Sem problemas bloqueantes.")
for p in problems:
    print(" -", p)

spark.createDataFrame(rep.astype({"fora_da_faixa": "float"})).write.mode("overwrite").saveAsTable(config.TABLE_QUALITY)

# COMMAND ----------

# Distribuição dos indicadores por ano
display(silver.groupby(config.YEAR_COL)[config.INDICATOR_COLS].describe().T)

# COMMAND ----------

spark.createDataFrame(silver).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_SILVER)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alvo e features (ano *t* → *t+1*)
# MAGIC
# MAGIC Features vêm do ano *t*; o alvo é observado em *t+1*. Estudantes sem registro em *t+1* não têm rótulo
# MAGIC observável e saem da base — a taxa de perda é medida abaixo porque pode introduzir viés (evasão).

# COMMAND ----------

for strategy in ("defasagem_next_year", "not_pv_next_year", "inde_drop_next_year"):
    try:
        ds = build_next_year_dataset(silver, strategy)
        print(f"{strategy:24s} n={len(ds):5d}  prevalência={ds[config.TARGET_COL].mean():.3f}  "
              f"por ano={ds.groupby(config.YEAR_COL)[config.TARGET_COL].mean().round(3).to_dict()}")
    except KeyError as e:
        print(f"{strategy:24s} indisponível (coluna ausente: {e})")

# COMMAND ----------

gold = build_next_year_dataset(silver, config.TARGET_STRATEGY)
total_t = silver[silver[config.YEAR_COL] < silver[config.YEAR_COL].max()].shape[0]
print(f"Pares com rótulo: {len(gold)} de {total_t} registros com ano seguinte possível "
      f"({len(gold) / total_t:.1%}).")
display(gold[config.TARGET_COL].value_counts())

spark.createDataFrame(gold).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.TABLE_FEATURES)
