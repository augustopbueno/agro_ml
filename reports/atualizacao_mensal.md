# Atualização mensal: CEPEA até 25/11/2026

Gerado em 26/11/2026 pelo workflow `atualizacao_mensal`.

## Dados

- Período usado nos modelos: julho/2016 a outubro/2026 (124 meses fechados)
- Soja: Indicador da Soja CEPEA/ESALQ, Paraná, média de outubro/2026 R$ 149,40/saca, último preço diário R$ 162,98 em 25/11/2026, média de 16 pregões de novembro/2026 R$ 156,80
- Milho: Indicador do Milho ESALQ/BM&FBovespa, média de outubro/2026 R$ 65,76/saca, último preço diário R$ 67,94 em 25/11/2026, média de 16 pregões de novembro/2026 R$ 67,22

## Previsão com faixa (próximos 6 meses)

As previsões partem do preço diário mais recente do CEPEA. Soja: o último preço diário, que no backtest não perdeu para nenhum modelo. Milho: Ridge direto por horizonte com as estimativas de estoque e as revisões do relatório WASDE do USDA, prevendo a variação a partir do preço diário. O mês em curso é estimado com a média dos pregões que já fecharam e o último preço repetido nos dias úteis que faltam. Faixa: quantil do erro das previsões passadas do mesmo horizonte, simuladas no mesmo dia do mês (conformal simétrica).

| Mês | Soja | Faixa 80% | Milho | Faixa 80% |
|---|---|---|---|---|
| novembro/2026 (em curso) | R$ 158,27 | R$ 157,73 a 158,80 | R$ 67,38 | R$ 67,17 a 67,58 |
| dezembro/2026 | R$ 162,98 | R$ 154,86 a 171,52 | R$ 68,99 | R$ 64,32 a 74,00 |
| janeiro/2027 | R$ 162,98 | R$ 147,27 a 180,37 | R$ 70,41 | R$ 61,87 a 80,13 |
| fevereiro/2027 | R$ 162,98 | R$ 143,18 a 185,52 | R$ 71,90 | R$ 62,53 a 82,68 |
| março/2027 | R$ 162,98 | R$ 138,78 a 191,40 | R$ 70,27 | R$ 59,94 a 82,37 |
| abril/2027 | R$ 162,98 | R$ 136,55 a 194,53 | R$ 67,72 | R$ 56,78 a 80,76 |

### Quanto a faixa acertou no passado

Percentual de vezes em que o preço real caiu dentro da faixa, em previsões feitas desde jan/2022 só com a informação disponível em cada data, no mesmo dia do mês desta rodada.

| Horizonte | Soja 80% | Soja 95% | Milho 80% | Milho 95% |
|---|---|---|---|---|
| Mês em curso | 81% | 93% | 91% | 98% |
| 1 mês depois | 88% | 98% | 93% | 96% |
| 2 meses depois | 91% | 96% | 87% | 96% |
| 3 meses depois | 87% | 100% | 89% | 94% |
| 4 meses depois | 85% | 100% | 89% | 92% |
| 5 meses depois | 90% | 100% | 88% | 92% |

## Validação walk-forward (últimos 12 meses, horizonte de 1 mês)

| Modelo | MAPE Soja | MAPE Milho |
|---|---|---|
| Ingênuo (repete a média do último mês) | 3.63% | 3.29% |
| Prophet | 6.39% | 4.35% |
| Random Forest | 2.74% | 2.81% |
| XGBoost | 2.59% | 3.02% |
| Modelo da faixa (soja: último preço diário; milho: Ridge + USDA a partir do preço diário)* | 2.37% | 1.97% |

\* Previsão feita no último dia útil do mês anterior, partindo do último preço diário. Os outros modelos partem da média do mês.

## Como foi o mês anterior

- Ainda não há previsão de rodada anterior para comparar. A comparação aparece a partir da próxima atualização.

## Rascunho de post para o LinkedIn

Revise antes de publicar. Imagem sugerida: `reports/figures/post_linkedin.png`

```text
Atualizei meu modelo de previsão de soja e milho com os preços do CEPEA até 25/11/2026.

Novembro/2026 ainda não fechou: nos 16 pregões até agora, a soja teve média de R$ 156,80 e o milho de R$ 67,22.

Para dezembro/2026: soja em R$ 162,98, com 80% de chance de ficar entre R$ 154,86 e R$ 171,52. Milho em R$ 68,99, entre R$ 64,32 e R$ 74,00.

Para fevereiro/2027 a faixa já abre: soja entre R$ 143 e R$ 186, milho entre R$ 63 e R$ 83. A incerteza cresce com o prazo, e acima de 6 meses a faixa fica larga demais para ser útil, então não publico.

Testei a faixa no passado: desde 2022, a de 80% para o mês seguinte pegou o preço real em 88% das vezes na soja e 93% no milho. A previsão parte do preço diário mais recente e, no milho, usa também as estimativas de estoque do USDA. Clima e custo de fertilizante ainda ficam de fora.

A atualização roda sozinha todo mês pelo GitHub Actions. Código e dados: github.com/augustopbueno/agro_ml

#dados #machinelearning #agronegocio #python
```
