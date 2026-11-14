"""Configuração central do pipeline.

Tudo que muda entre ambientes (catálogo, schema, nomes de tabela/modelo) e as
decisões que precisam ser fixadas ANTES do treino (alvo, features, quality gate)
fica aqui, versionado no Git. Nenhum segredo ou token deve ser colocado neste arquivo.
"""

# ---------------------------------------------------------------------------
# Unity Catalog (Free Edition cria por padrão o catálogo "workspace")
# ---------------------------------------------------------------------------
CATALOG = "workspace"
SCHEMA = "passos_magicos"
VOLUME = "raw"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
RAW_FILE_NAME = "base_pede.xlsx"  # ajustar ao nome real do arquivo enviado ao Volume

TABLE_BRONZE = f"{CATALOG}.{SCHEMA}.bronze_pede"
TABLE_SILVER = f"{CATALOG}.{SCHEMA}.silver_pede"
TABLE_FEATURES = f"{CATALOG}.{SCHEMA}.gold_features"
TABLE_QUALITY = f"{CATALOG}.{SCHEMA}.data_quality_report"
TABLE_INFERENCE_LOG = f"{CATALOG}.{SCHEMA}.inference_log"

# ---------------------------------------------------------------------------
# MLflow / registro
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "/Shared/passos_magicos_risco"  # ajustar para /Users/<seu_usuario>/... se preferir
MODEL_NAME = f"{CATALOG}.{SCHEMA}.risco_acompanhamento"
CHAMPION_ALIAS = "champion"
SERVING_ENDPOINT_NAME = "passos-magicos-risco"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Problema de ML
# ---------------------------------------------------------------------------
# Unidade de análise: estudante-ano.
# Momento da previsão: fechamento da avaliação do ano t (indicadores do ano t já
# calculados). O alvo é observado no ano t+1. Assim nenhuma feature depende de
# informação posterior ao momento da previsão.
#
# Alvo padrão (hipótese a validar no notebook 01): o estudante NÃO atinge o ponto
# de virada no ano t+1 E/OU apresenta defasagem. Ver features.build_target().
# A definição final deve ser confirmada após a EDA e documentada no model_card.
TARGET_COL = "target"
TARGET_STRATEGY = "defasagem_next_year"  # opções em features.TARGET_STRATEGIES

# Colunas canônicas da camada silver (formato longo: 1 linha por estudante-ano)
ID_COL = "student_id"  # hash do RA: usado só para juntar anos e agrupar validação
YEAR_COL = "ano"

INDICATOR_COLS = ["inde", "iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]
NUMERIC_FEATURES = INDICATOR_COLS + ["fase", "idade", "anos_pm", "defasagem"]

# Colunas que NUNCA entram como feature (identificadores, sensíveis ou derivadas do alvo)
EXCLUDED_FROM_FEATURES = [
    "nome", "ra", ID_COL, "genero", "instituicao_ensino", "turma",
    "avaliador", "rec_psicologia", "indicado",
]

# Faixas plausíveis para validação de dados e de entrada do modelo
INDICATOR_RANGE = (0.0, 10.0)
FEATURE_RANGES = {
    **{c: INDICATOR_RANGE for c in INDICATOR_COLS},
    "fase": (0, 9),
    "idade": (5, 30),
    "anos_pm": (0, 15),
    "defasagem": (-9, 9),
}

# ---------------------------------------------------------------------------
# Quality gate — definido ANTES do treino (Requisito 6)
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "pr_auc"  # classe positiva minoritária; custo maior em falso negativo
QUALITY_GATE = {
    # candidato precisa superar o melhor baseline na métrica principal por esta margem
    "min_gain_over_baseline_pr_auc": 0.02,
    # piso absoluto de recall no limiar operacional (não deixar passar quem precisa de apoio)
    "min_recall": 0.60,
    # piso de precisão para não sobrecarregar a equipe pedagógica com falsos alarmes
    "min_precision": 0.30,
    # calibração mínima aceitável
    "max_brier": 0.25,
}
TARGET_RECALL_FOR_THRESHOLD = 0.70  # limiar escolhido na validação cruzada do treino
