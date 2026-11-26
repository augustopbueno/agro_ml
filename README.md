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
| CEPEA/ESALQ | Indicador da Soja CEPEA/ESALQ (Paraná) | Mensal |
| CEPEA/ESALQ | Indicador do Milho ESALQ/BM&FBovespa | Mensal |
| Banco Central (SGS) | Câmbio USD/BRL, Selic, IPCA | Diária/Mensal |
| USDA (relatório WASDE) | Estoque/consumo de soja e milho (mundo e EUA) e revisões de produção do Brasil, como publicados em cada relatório | Mensal |

<!-- PERIODO:INICIO -->
Período usado nos modelos: julho/2016 a outubro/2026.
<!-- PERIODO:FIM -->

O CEPEA não tem API pública. As planilhas são baixadas manualmente pelo portal de consultas (detalhes no notebook 01). O resto da atualização é automático, veja a seção "Atualização mensal automática".

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
Atualizado com o CEPEA de outubro/2026.

**Previsão com faixa de 80%**

| Mês | Soja | Faixa 80% | Milho | Faixa 80% |
|---|---|---|---|---|
| novembro/2026 | R$ 149,90 | R$ 140,60 a 159,83 | R$ 66,62 | R$ 61,45 a 72,23 |
| janeiro/2027 | R$ 149,39 | R$ 131,24 a 170,05 | R$ 69,87 | R$ 60,01 a 81,35 |
| abril/2027 | R$ 150,03 | R$ 124,29 a 181,11 | R$ 67,31 | R$ 55,94 a 80,98 |

Desde 2022, a faixa de 80% do mês seguinte pegou o preço real em 86% das vezes na soja e 89% no milho.

**Validação walk-forward, últimos 12 meses, horizonte de 1 mês**

| Modelo | MAPE Soja | MAPE Milho |
|---|---|---|
| Ingênuo (repete o último preço) | 3.63% | 3.30% |
| Prophet | 6.39% | 4.42% |
| Random Forest | 2.74% | 2.84% |
| XGBoost | 2.59% | 2.99% |
| Modelo da faixa (soja: média ingênuo + Ridge; milho: Ridge + USDA) | 3.40% | 3.38% |

Detalhes, faixa de 95% e histórico em [reports/atualizacao_mensal.md](reports/atualizacao_mensal.md).
<!-- RESULTADOS:FIM -->

A diferença entre os modelos com lag (Random Forest e XGBoost) e o Prophet é esperada: os modelos com lag capturam autocorrelação forte da série (o preço do mês anterior explica mais de 70% da variância), enquanto o Prophet foca em tendência e sazonalidade sem usar esse sinal.

A previsão futura do Random Forest é mais conservadora que a do Prophet porque o forecast recursivo tende a convergir para a média. Para horizontes longos, o Prophet é mais informativo. Para o mês seguinte, o Random Forest é mais preciso.

---

## Atualização mensal automática

O workflow `.github/workflows/atualizacao_mensal.yml` roda o `src/atualizar.py`, que refaz o pipeline dos notebooks 01, 03, 04 e 05 de ponta a ponta: atualiza câmbio, Selic e IPCA pela API do Banco Central, recalcula as features, refaz a validação walk-forward, gera a previsão com faixa de 80% e 95% para até 6 meses e as figuras, compara a previsão do mês anterior com o valor real e atualiza este README.

A faixa vem de um backtest com dezenas de origens desde 2019. Na soja, o ponto é a média entre o ingênuo (repete o último preço) e um Ridge direto por horizonte. No milho, é um Ridge direto que também usa as estimativas de estoque e as revisões do relatório WASDE do USDA: no backtest, isso baixou o erro em 6 meses de cerca de 18% para 14% e deixou a faixa de 80% cerca de 35% mais estreita, nos dois subperíodos testados (2022-23 e 2024-25). Na soja, os dados do USDA não ajudaram. A largura da faixa é o quantil do erro das previsões passadas do mesmo horizonte (conformal simétrica). O intervalo próprio do Prophet foi descartado para isso porque, no mesmo teste, era largo e pegava o preço real só em cerca de metade das vezes.

O relatório WASDE sai por volta do dia 10 de cada mês, e o USDA guarda os dados de cada relatório como foram publicados. O histórico desde 2016 está em `data/raw/wasde_features.csv` e o workflow baixa os relatórios novos sozinho. São dados do governo americano, de domínio público.

Ele roda em três situações:

- **Dia 15 de cada mês:** se a planilha do CEPEA do mês anterior ainda não estiver no repositório, abre uma issue lembrando de subir.
- **Quando as planilhas do CEPEA são atualizadas** em `data/raw/` (upload pelo próprio GitHub).
- **Manualmente**, pela aba Actions.

Para atualizar o CEPEA: baixar as séries mensais de soja (Paraná) e milho no [portal de consultas](https://www.cepea.org.br/br/consultas-ao-banco-de-dados-do-site.aspx), de 01/01/2016 até o fim do mês anterior, renomear para `cepea_soja.xls` e `cepea_milho.xls` (sem converter) e subir em `data/raw/` com "Add file > Upload files". O site tem verificação anti-robô, por isso essa parte não dá para automatizar.

Para rodar local: `pip install -r requirements.txt` e `python src/atualizar.py` (ou `--sem-bcb` para não consultar o Banco Central).

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
