# Plano de monitoramento e governança

## Métricas operacionais
| Sinal | Fonte | Limite / ação |
|---|---|---|
| Latência e taxa de erro do endpoint | Model Serving (métricas do endpoint) | erro > 5% → investigar |
| % de entradas rejeitadas (`status = erro`) | `inference_log` | > 10% → revisar integração do cliente |
| Volume de chamadas por dia | `inference_log` | queda abrupta → verificar consumidor |

## Qualidade dos dados de entrada / drift
| Sinal | Como | Limite |
|---|---|---|
| PSI por feature (referência = treino) | `src/monitoring.drift_report` | > 0,1 atenção · > 0,2 alerta |
| % nulos por feature | idem | aumento > 10 p.p. |
| Distribuição das probabilidades e % `sinal_apoio = sim` | `inference_log` | variação > 2× a prevalência do treino |

## Qualidade do modelo
- Quando o ano *t+1* fechar, juntar os rótulos reais às previsões do `inference_log` e recalcular PR-AUC,
  recall e precisão (o rótulo chega com ~1 ano de atraso).
- Reavaliar/re-treinar se a PR-AUC cair abaixo do melhor baseline + 0,02 ou o recall cair abaixo de 0,60.
- Todo re-treino passa novamente pelo quality gate; o alias `@champion` só muda se aprovado.

## Segurança e privacidade
- RA pseudonimizado (hash) e nome removido na camada silver; nenhum dado pessoal no GitHub.
- Tokens apenas via OAuth/variáveis de ambiente/secrets da Databricks.
- Acesso às tabelas e ao endpoint restrito via permissões do Unity Catalog.

## Limites de uso
- Previsão é sinal de apoio, não decisão. Revisão humana obrigatória antes de qualquer ação.
- Não usar para ranqueamento público, punição ou seleção de estudantes.
