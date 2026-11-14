# Arquitetura

```
Base PEDE (xlsx) ──upload──▶ UC Volume  workspace.passos_magicos.raw
                                  │
             notebook 01          ▼
                         bronze_pede  (bruto, todas as abas, como texto)
                                  │  padronização, tipos, RA → hash, remove nome
                                  ▼
                         silver_pede  (1 linha por estudante-ano)  ──▶ data_quality_report
                                  │  join t → t+1, alvo observado em t+1
                                  ▼
                         gold_features
                                  │
             notebook 02          ▼
              MLflow experiment: baselines + candidatos (CV agrupada por estudante no treino)
                                  │  teste temporal (último ano) + quality gate
                                  ▼
             notebook 03   UC Model Registry  workspace.passos_magicos.risco_acompanhamento
                                  │  alias @champion só se gate aprovado
                                  ▼
                   Model Serving endpoint  passos-magicos-risco   (ou fallback pyfunc)
                                  │
                                  ▼
            cliente (notebook 03 / scripts/client_rest.py) ──▶ inference_log + drift (PSI)
```

## Decisões

| Decisão | Motivo |
|---|---|
| Lógica em `src/`, notebooks só orquestram | Testável localmente (pytest) e versionado no Git |
| Divisão temporal (treino passado → teste último ano) | Simula o uso real e evita resultado otimista |
| CV agrupada por estudante no treino | O mesmo estudante aparece em anos diferentes |
| Modelo logado como `pyfunc` com validação embutida | Endpoint e fallback aplicam as mesmas regras de entrada e retornam erro por linha |
| Alias `champion` condicionado ao gate | Promoção não depende de escolha manual |

## Limitações do ambiente (preencher após verificar o workspace)

- Unity Catalog: disponível (catálogo `workspace`)
- Model Serving: disponível no menu; confirmação final na criação do endpoint (notebook 03)
- Compute: somente serverless
- Jobs agendados: _a verificar_
