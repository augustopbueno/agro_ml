# Previsão de Preço de Commodities Agrícolas

Estudo pessoal de Machine Learning aplicado ao agronegócio brasileiro.

O projeto aplica análises estatísticas e modelos preditivos para estimar as variações de preço de soja e milho nos próximos dois anos. O ponto de partida foi uma inquietação prática: depois dos choques que o mercado sofreu com a pandemia e a guerra na Ucrânia, como construir uma análise que seja honesta com a incerteza sem perder utilidade?

A resposta foi combinar modelos de tendência de longo prazo com modelos que aprendem com o comportamento recente dos preços, e incluir sazonalidade ligada ao calendário de safra brasileiro para manter a previsão o mais aderente possível ao que o mercado realmente faz.

---

## Contexto

Esse projeto nasceu da vontade de entender o que está por baixo, aprender scikit-learn na prática e construir algo que eu pudesse defender com dados.

---

## Dados

| Fonte | Série | Periodicidade |
|---|---|---|
| CEPEA/ESALQ | Preço soja e milho (Paraná) | Mensal |
| Banco Central (SGS) | Câmbio USD/BRL, Selic, IPCA | Diária/Mensal |

Período: outubro de 2016 a maio de 2026.

O CEPEA não tem API pública. Os dados foram baixados manualmente pelo portal de consultas e estão documentados no notebook 01.

---

## Estrutura do projeto

```
agro_ml/
├── data/
│   ├── raw/          # dados brutos (CEPEA e BCB)
│   └── processed/    # features prontas para modelagem
├── notebooks/
│   ├── 01_coleta_dados.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelo_prophet.ipynb
│   └── 05_modelo_sklearn.ipynb
├── reports/
│   └── figures/      # gráficos gerados pelos notebooks
├── requirements.txt
└── README.md
```

---

## Metodologia

**Análise exploratória (notebook 02)**

Antes de qualquer modelo, entender o comportamento da série. O foco foi sazonalidade ligada ao ciclo de safra brasileiro, correlação com câmbio e identificação de períodos atípicos. A decomposição STL separou tendência, sazonalidade e resíduo, o que ajudou a entender quanto do movimento de preço é estrutural e quanto é ruído.

**Feature engineering (notebook 03)**

As features foram construídas a partir de três fontes de informação: o comportamento recente do próprio preço (lags t-1, t-2, t-3), o câmbio defasado (o efeito cambial demora para chegar ao preço interno) e o calendário agrícola (mês, trimestre, flag de colheita). O preço também foi deflacionado pelo IPCA para separar inflação geral de valorização real da commodity.

**Modelos (notebooks 04 e 05)**

Dois modelos com abordagens complementares:

- **Prophet** - modelo de tendência e sazonalidade, sem features externas. Bom para capturar o comportamento de longo prazo e gerar intervalo de confiança. Horizonte de 24 meses com incerteza crescente, o que reflete a realidade da série.
- **Random Forest e XGBoost** - modelos supervisionados que aprendem com as features de lag. Capturam melhor o comportamento de curto prazo.

A validação usou walk-forward em todos os modelos, treinando no histórico até t e prevendo t+1, para simular o uso real e evitar data leakage.

---

## Resultados

Validação walk-forward, últimos 12 meses, horizonte de 1 mês.

| Modelo | MAPE Soja | MAPE Milho |
|---|---|---|
| Prophet | 7.65% | 9.57% |
| Random Forest | 1.82% | 2.77% |
| XGBoost | 1.82% | 3.12% |

O Random Forest teve o melhor desempenho em ambas as commodities. A diferença para o Prophet é esperada: os modelos com lag capturam autocorrelação forte da série (o preço do mês anterior explica mais de 70% da variância), enquanto o Prophet foca em tendência e sazonalidade sem usar esse sinal.

A previsão futura do Random Forest é mais conservadora que a do Prophet porque o forecast recursivo tende a convergir para a média. Para horizontes longos, o Prophet é mais informativo. Para o mês seguinte, o Random Forest é mais preciso.

---

## Limitações e próximos passos

Alguns fatores com impacto real no preço das commodities não estão capturados nos modelos atuais:

**Geopolítica e custo de insumos**

Conflitos como a guerra na Ucrânia em 2022 e as tensões atuais no Oriente Médio afetam o mercado de formas distintas para cada commodity. No caso do milho, o impacto é mais direto: fertilizantes como ureia, MAP e potássio têm produção concentrada em regiões afetadas por esses conflitos. Quando o custo de produção sobe, o preço do grão acompanha. Para a soja, o efeito é mais indireto, passando principalmente pelo câmbio e pelo redirecionamento de fluxos comerciais globais.

O câmbio defasado já captura parte desse efeito indiretamente, mas eventos geopolíticos pontuais não têm série histórica previsível. A abordagem mais honesta é documentá-los como limitação e monitorar variáveis proxy como preço de fertilizantes (ANDA) e frete marítimo (Baltic Dry Index) para expansões futuras do modelo.

**Clima**

El Nino e La Nina afetam diretamente a produtividade das safras brasileiras e, por consequência, o preço. A próxima versão do projeto vai incluir o índice ONI (Oceanic Nino Index) como variável exógena. Dados do INMET para temperatura e precipitação no Paraná também estão no radar.

**Granularidade**

Os dados do CEPEA estão disponíveis apenas em periodicidade mensal pelo portal de consultas. Para análises de curto prazo com granularidade diária, seria necessário acesso a fontes privadas.


---

## Stack

Python, pandas, scikit-learn, Prophet, XGBoost, statsmodels, matplotlib, seaborn

---

## Autor

Augusto Paludo Bueno
Analista de BI
[LinkedIn](https://linkedin.com/in/augustopbueno) | [GitHub](https://github.com/augustopbueno)
