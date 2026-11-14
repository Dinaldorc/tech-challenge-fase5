# Model card — risco_acompanhamento

> Preencher com os resultados reais após rodar os notebooks 01–03.

## Objetivo
Sinalizar estudantes que **talvez** precisem de acompanhamento adicional no próximo ciclo, para apoiar a
investigação da equipe pedagógica. Não substitui avaliação humana.

## Definição do problema (Requisito 1)
| Item | Definição |
|---|---|
| Unidade de análise | estudante-ano |
| População | estudantes avaliados no ano *t* que também aparecem no ano *t+1* |
| Momento da previsão | após o fechamento dos indicadores do ano *t* |
| Variável-alvo | _a confirmar após EDA_ (`config.TARGET_STRATEGY`) |
| Features | indicadores do ano *t* (INDE, IAA, IEG, IPS, IPP, IDA, IPV, IAN), fase, idade, anos na PM, defasagem |
| Excluídas | nome, RA, gênero, instituição, turma, avaliadores, recomendações (identificadores, sensíveis ou pós-fato) |

## Dados
- Anos: _preencher_ · Registros: _preencher_ · Prevalência do alvo: _preencher_
- Perda por ausência em *t+1*: _preencher_ (possível viés de evasão)

## Validação
- Treino: pares com origem em _anos_ · Teste: origem em _ano_
- Seleção de modelo e limiar: CV de 5 folds agrupada por estudante, só no treino

## Resultados
| Modelo | PR-AUC | ROC-AUC | Recall | Precisão | F1 | Brier |
|---|---|---|---|---|---|---|
| baseline_prior | | | | | | |
| baseline_inde | | | | | | |
| candidato | | | | | | |

## Quality gate (definido antes do treino — `src/config.py`)
| Critério | Limite | Resultado |
|---|---|---|
| Ganho de PR-AUC sobre o melhor baseline | ≥ 0,02 | |
| Recall no limiar | ≥ 0,60 | |
| Precisão no limiar | ≥ 0,30 | |
| Brier | ≤ 0,25 | |

**Decisão:** _promovido / rejeitado_ — justificativa.

## Limiar e custo dos erros
- **Falso negativo:** estudante que precisava de apoio não é sinalizado → custo maior.
- **Falso positivo:** equipe revisa um caso sem necessidade → custo de tempo.
- O limiar é o maior valor que mantém recall ≥ 0,70 na CV do treino.

## Registro
- Nome: `workspace.passos_magicos.risco_acompanhamento` · Versão: _n_ · Alias: `@champion`
- Entrada: features numéricas (nulos permitidos; faixas em `config.FEATURE_RANGES`)
- Saída: `status`, `erros`, `probabilidade`, `sinal_apoio`, `faixa`, `limiar`, `aviso`

## Uso responsável e limitações
- Correlação ≠ causalidade; o modelo não explica por que um estudante tem risco.
- Saída sempre acompanhada de aviso; nunca usar isoladamente para decidir sobre um estudante.
- Base pequena e de uma única organização; desempenho pode variar entre fases.
