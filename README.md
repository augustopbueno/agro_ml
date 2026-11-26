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
| CEPEA/ESALQ | Indicador da Soja CEPEA/ESALQ (Paraná) | Diária |
| CEPEA/ESALQ | Indicador do Milho ESALQ/BM&FBovespa | Diária |
| Banco Central (SGS) | Câmbio USD/BRL, Selic, IPCA | Diária/Mensal |
| USDA (relatório WASDE) | Estoque/consumo de soja e milho (mundo e EUA) e revisões de produção do Brasil, como publicados em cada relatório | Mensal |

<!-- PERIODO:INICIO -->
Período usado nos modelos: julho/2016 a outubro/2026. Preço diário até 25/11/2026.
<!-- PERIODO:FIM -->

O CEPEA não tem API pública. As planilhas são baixadas manualmente pelo portal de consultas, na periodicidade diária. Os modelos mensais usam a média de cada mês fechado, e as previsões partem do preço diário mais recente. O resto da atualização é automático, veja a seção "Atualização mensal automática".

Licença dos dados do CEPEA: CC BY-NC 4.0 (uso não comercial, citando a fonte).

---

## Estrutura do projeto

```
agro_ml/
├── data/
│   ├── raw/          # dados brutos (CEPEA, BCB e USDA)
│   └── processed/    # features prontas para modelagem
├── notebooks/
│   ├── 01_coleta_dados.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelo_prophet.ipynb
│   └── 05_modelo_sklearn.ipynb
├── reports/
│   ├── figures/                  # gráficos gerados pelos notebooks e pela atualização mensal
│   ├── atualizacao_mensal.md     # resultado da última atualização + rascunho de post
│   └── historico_previsoes.csv   # previsão de cada mês, para medir o erro depois
├── src/
│   └── atualizar.py  # pipeline completo (BCB, CEPEA, features, modelos, figuras, README)
├── .github/workflows/
│   └── atualizacao_mensal.yml
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

<!-- RESULTADOS:INICIO -->
Atualizado com o CEPEA até 25/11/2026.

**Previsão com faixa de 80%**

| Mês | Soja | Faixa 80% | Milho | Faixa 80% |
|---|---|---|---|---|
| novembro/2026 (em curso) | R$ 158,27 | R$ 157,73 a 158,80 | R$ 67,38 | R$ 67,17 a 67,58 |
| dezembro/2026 | R$ 162,98 | R$ 154,86 a 171,52 | R$ 68,99 | R$ 64,32 a 74,00 |
| janeiro/2027 | R$ 162,98 | R$ 147,27 a 180,37 | R$ 70,41 | R$ 61,87 a 80,13 |
| abril/2027 | R$ 162,98 | R$ 136,55 a 194,53 | R$ 67,72 | R$ 56,78 a 80,76 |

Desde 2022, simulando a previsão no mesmo dia do mês, a faixa de 80% pegou o preço real em 81% das vezes na soja e 91% no milho para o mês em curso, e em 88% e 93% para o mês seguinte.

**Validação walk-forward, últimos 12 meses, horizonte de 1 mês**

| Modelo | MAPE Soja | MAPE Milho |
|---|---|---|
| Ingênuo (repete a média do último mês) | 3.63% | 3.29% |
| Prophet | 6.39% | 4.35% |
| Random Forest | 2.74% | 2.81% |
| XGBoost | 2.59% | 3.02% |
| Modelo da faixa (soja: último preço diário; milho: Ridge + USDA a partir do preço diário)* | 2.37% | 1.97% |

\* Previsão feita no último dia útil do mês anterior, partindo do último preço diário. Os outros modelos partem da média do mês.

Detalhes, faixa de 95% e histórico em [reports/atualizacao_mensal.md](reports/atualizacao_mensal.md).
<!-- RESULTADOS:FIM -->

A diferença entre os modelos com lag (Random Forest e XGBoost) e o Prophet é esperada: os modelos com lag capturam autocorrelação forte da série (o preço do mês anterior explica mais de 70% da variância), enquanto o Prophet foca em tendência e sazonalidade sem usar esse sinal.

A previsão futura do Random Forest é mais conservadora que a do Prophet porque o forecast recursivo tende a convergir para a média. Para horizontes longos, o Prophet é mais informativo. Para o mês seguinte, o Random Forest é mais preciso.

---

## Atualização mensal automática

O workflow `.github/workflows/atualizacao_mensal.yml` roda o `src/atualizar.py`, que refaz o pipeline dos notebooks 01, 03, 04 e 05 de ponta a ponta: atualiza câmbio, Selic e IPCA pela API do Banco Central, recalcula as features, refaz a validação walk-forward, gera a previsão com faixa de 80% e 95% para até 6 meses e as figuras, compara a previsão do mês anterior com o valor real e atualiza este README.

A faixa vem de um backtest com dezenas de origens desde 2019. As previsões partem do preço diário mais recente do CEPEA, e não da média do mês fechado: quando a atualização roda no dia 15, metade do mês já fechou e a média do mês anterior está defasada. O mês em curso é estimado com a média dos pregões que já fecharam mais o último preço para os dias que faltam.

Na soja, a previsão é o próprio último preço diário. Nenhum modelo testado bateu isso de forma consistente, nem os dados do USDA, nem modelos pré-treinados de séries temporais (Chronos e TimesFM). No milho, é um Ridge direto por horizonte que usa as estimativas de estoque e as revisões do relatório WASDE do USDA e prevê a variação a partir do preço diário.

No backtest de 2022 a 2025, com a previsão feita no dia 15, usar o preço diário baixou o erro médio assim:

| Horizonte | Soja antes | Soja agora | Milho antes | Milho agora |
|---|---|---|---|---|
| Mês em curso | 3,3% | 0,9% | 4,0% | 0,9% |
| 1 mês depois | 5,6% | 3,8% | 7,5% | 4,8% |
| 2 meses depois | 7,1% | 5,9% | 9,7% | 7,7% |
| 5 meses depois | 12,0% | 10,0% | 13,8% | 12,8% |

A melhora aparece em todos os anos de 2022 a 2025 nos prazos curtos. Em 5 meses ela é pequena, porque ali o que pesa é safra e câmbio.

A largura da faixa é o quantil do erro das previsões passadas do mesmo horizonte (conformal simétrica), simuladas no mesmo dia do mês da rodada. Por isso a faixa do mês em curso fica estreita no fim do mês e mais larga no começo. O intervalo próprio do Prophet foi descartado para isso porque, no mesmo teste, era largo e pegava o preço real só em cerca de metade das vezes.

O relatório WASDE sai por volta do dia 10 de cada mês, e o USDA guarda os dados de cada relatório como foram publicados. O histórico desde 2016 está em `data/raw/wasde_features.csv` e o workflow baixa os relatórios novos sozinho. São dados do governo americano, de domínio público.

Ele roda em três situações:

- **Dia 15 de cada mês:** refaz tudo com o que está no repositório e, se a planilha do CEPEA não tiver o mês anterior completo, abre uma issue lembrando de subir.
- **Quando as planilhas do CEPEA são atualizadas** em `data/raw/` (upload pelo próprio GitHub).
- **Manualmente**, pela aba Actions.

Para atualizar o CEPEA: no [portal de consultas](https://www.cepea.org.br/br/consultas-ao-banco-de-dados-do-site.aspx), baixar soja (Indicador da Soja CEPEA/ESALQ, Paraná) e milho (Indicador do Milho ESALQ/BM&FBovespa) com periodicidade **Diário**, de 01/01/2016 até hoje. Renomear para `cepea_soja.xls` e `cepea_milho.xls` (sem converter) e subir em `data/raw/` com "Add file > Upload files". Quanto mais perto do fim do mês, mais o mês em curso já está fechado e mais precisa fica a previsão. O site tem verificação anti-robô, por isso essa parte não dá para automatizar. A planilha mensal ainda funciona, mas aí as previsões voltam ao método antigo, partindo da média do mês.

Para rodar local: `pip install -r requirements.txt` e `python src/atualizar.py` (ou `--sem-bcb` para não consultar o Banco Central).

---

## Limitações e próximos passos

Alguns fatores com impacto real no preço das commodities não estão capturados nos modelos atuais:

**Geopolítica e custo de insumos**

Conflitos como a guerra na Ucrânia em 2022 e as tensões atuais no Oriente Médio afetam o mercado de formas distintas para cada commodity. No caso do milho, o impacto é mais direto: fertilizantes como ureia, MAP e potássio têm produção concentrada em regiões afetadas por esses conflitos. Quando o custo de produção sobe, o preço do grão acompanha. Para a soja, o efeito é mais indireto, passando principalmente pelo câmbio e pelo redirecionamento de fluxos comerciais globais.

O câmbio defasado já captura parte desse efeito indiretamente, mas eventos geopolíticos pontuais não têm série histórica previsível. A abordagem mais honesta é documentá-los como limitação e monitorar variáveis proxy como preço de fertilizantes (ANDA) e frete marítimo (Baltic Dry Index) para expansões futuras do modelo.

**Clima**

El Nino e La Nina afetam diretamente a produtividade das safras brasileiras e, por consequência, o preço. A próxima versão do projeto vai incluir o índice ONI (Oceanic Nino Index) como variável exógena. Dados do INMET para temperatura e precipitação no Paraná também estão no radar.

**Futuros**

Os contratos futuros de soja e milho (B3 e Chicago) trariam a expectativa do mercado para os próximos meses, mas não encontrei fonte gratuita que permita coleta automatizada. A B3 proíbe expressamente a coleta automática dos dados do site.

---

## Stack

Python, pandas, scikit-learn, Prophet, XGBoost, statsmodels, matplotlib, seaborn

---

## Autor

Augusto Paludo Bueno
Analista de BI
[LinkedIn](https://linkedin.com/in/augustopbueno) | [GitHub](https://github.com/augustopbueno)
