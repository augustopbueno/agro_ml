# Atualização mensal: outubro/2026

Gerado em 26/11/2026 pelo workflow `atualizacao_mensal`.

## Dados

- Período usado nos modelos: julho/2016 a outubro/2026 (124 meses)
- Soja: Indicador da Soja CEPEA/ESALQ, Paraná, último valor R$ 149,40/saca
- Milho: Indicador do Milho ESALQ/BM&FBovespa, último valor R$ 65,76/saca

## Previsão com faixa (próximos 6 meses)

Soja: média entre o ingênuo (repete o último preço) e um Ridge direto por horizonte. Milho: Ridge direto por horizonte com as estimativas de estoque e as revisões do relatório WASDE do USDA. Faixa: quantil do erro das previsões passadas do mesmo horizonte (conformal simétrica).

| Mês | Soja | Faixa 80% | Milho | Faixa 80% |
|---|---|---|---|---|
| novembro/2026 | R$ 149,90 | R$ 140,60 a 159,83 | R$ 66,62 | R$ 61,45 a 72,23 |
| dezembro/2026 | R$ 149,83 | R$ 134,52 a 166,90 | R$ 68,37 | R$ 59,21 a 78,94 |
| janeiro/2027 | R$ 149,39 | R$ 131,24 a 170,05 | R$ 69,87 | R$ 60,01 a 81,35 |
| fevereiro/2027 | R$ 149,17 | R$ 126,11 a 176,46 | R$ 71,42 | R$ 60,19 a 84,74 |
| março/2027 | R$ 149,38 | R$ 126,11 a 176,94 | R$ 69,88 | R$ 58,67 a 83,25 |
| abril/2027 | R$ 150,03 | R$ 124,29 a 181,11 | R$ 67,31 | R$ 55,94 a 80,98 |

### Quanto a faixa acertou no passado

Percentual de vezes em que o preço real caiu dentro da faixa, em previsões feitas desde jan/2022 só com a informação disponível em cada data.

| Horizonte | Soja 80% | Soja 95% | Milho 80% | Milho 95% |
|---|---|---|---|---|
| 1 mês | 86% | 98% | 89% | 98% |
| 2 meses | 88% | 95% | 89% | 95% |
| 3 meses | 85% | 100% | 87% | 93% |
| 4 meses | 91% | 96% | 85% | 93% |
| 5 meses | 91% | 100% | 85% | 91% |
| 6 meses | 88% | 100% | 87% | 90% |

## Validação walk-forward (últimos 12 meses, horizonte de 1 mês)

| Modelo | MAPE Soja | MAPE Milho |
|---|---|---|
| Ingênuo (repete o último preço) | 3.63% | 3.30% |
| Prophet | 6.39% | 4.42% |
| Random Forest | 2.74% | 2.84% |
| XGBoost | 2.59% | 2.99% |
| Modelo da faixa (soja: média ingênuo + Ridge; milho: Ridge + USDA) | 3.40% | 3.38% |

## Como foi o mês anterior

- Ainda não há previsão de rodada anterior para comparar. A comparação aparece a partir da próxima atualização.

## Rascunho de post para o LinkedIn

Revise antes de publicar. Imagem sugerida: `reports/figures/post_linkedin.png`

```text
Atualizei meu modelo de previsão de soja e milho com os dados do CEPEA de outubro/2026.

Para novembro/2026: soja em R$ 149,90, com 80% de chance de ficar entre R$ 140,60 e R$ 159,83. Milho em R$ 66,62, entre R$ 61,45 e R$ 72,23.

Para janeiro/2027 a faixa já abre: soja entre R$ 131 e R$ 170, milho entre R$ 60 e R$ 81. A incerteza cresce com o prazo, e acima de 6 meses a faixa fica larga demais para ser útil, então não publico.

Testei a faixa no passado: desde 2022, a de 80% para o mês seguinte pegou o preço real em 86% das vezes na soja e 89% no milho. O modelo usa preço passado e câmbio e, no milho, as estimativas de estoque do USDA, que ajudaram a estreitar a faixa nos prazos maiores. Clima e custo de fertilizante ainda ficam de fora.

A atualização roda sozinha todo mês pelo GitHub Actions. Código e dados: github.com/augustopbueno/agro_ml

#dados #machinelearning #agronegocio #python
```
