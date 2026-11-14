# Passos Mágicos — pipeline de MLOps na Databricks

Datathon Fase 5 (Pós Tech FIAP). Pipeline reproduzível que estima a necessidade de acompanhamento adicional
de estudantes da Associação Passos Mágicos: **dados → features → treino → avaliação → registro → deploy →
inferência → monitoramento**. Roteiro do desafio em [docs/](docs/Datathon_Fase_5_Databricks_Passos_Magicos_AI_Scientist.pdf).

> As previsões são **sinais de apoio à investigação pedagógica**, nunca rótulos definitivos sobre um estudante.

## Estrutura

```
notebooks/
  01_data_quality_and_features.py   bronze → silver → gold, verificações de qualidade
  02_train_and_evaluate.py          baselines + candidatos, MLflow, quality gate
  03_register_and_inference.py      registro UC, @champion, Model Serving, inferência, drift
src/
  config.py      nomes de tabelas/modelo, alvo, features, critérios do quality gate
  data.py        ingestão, padronização, pseudonimização, relatório de qualidade
  features.py    alvo t → t+1, pré-processamento, divisão temporal
  train.py       modelos, CV agrupada, limiar, métricas, quality gate
  predict.py     validação de entrada + wrapper pyfunc servido
  monitoring.py  PSI / drift
scripts/client_rest.py   cliente REST do endpoint
tests/                   pytest (validação de entrada, alvo sem vazamento, gate)
docs/                    architecture.md, model_card.md, monitoring_plan.md
```

## Como reproduzir

### 1. Databricks (execução oficial)
1. Crie a conta no [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. **Workspace → Create → Git folder** apontando para este repositório.
3. Rode o notebook 01 uma vez para criar o schema e o Volume; envie a base PEDE para
   `/Volumes/workspace/passos_magicos/raw/` e ajuste `RAW_FILE_NAME` em `src/config.py`.
4. Rode os notebooks 01, 02 e 03 em ordem (compute serverless).

### 2. Local (VS Code) — desenvolvimento e testes
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pytest -q
```

### 3. Cliente REST
```bash
set DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
set DATABRICKS_TOKEN=<token>      # nunca versionar
python scripts/client_rest.py
```

## Recursos do ambiente e limitações

| Recurso | Status no workspace |
|---|---|
| Unity Catalog | Disponível (Free Edition, catálogo `workspace`) |
| Model Serving | Disponível no menu (CPU, sem GPU); criação do endpoint confirmada no notebook 03. Se falhar, o notebook usa `mlflow.pyfunc.load_model` (fallback; **não** é deploy em endpoint) |
| Jobs | _a verificar_ |
| Compute | Apenas serverless (Free Edition) |

## Privacidade
Nenhum dado da base está no repositório. O RA é convertido em hash e o nome é removido na camada silver.
Credenciais apenas via OAuth / variáveis de ambiente / secrets.
