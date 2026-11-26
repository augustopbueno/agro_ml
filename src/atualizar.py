"""
Atualização mensal do agro_ml.

Roda de ponta a ponta o que os notebooks 01, 03, 04 e 05 fazem:
  1. Atualiza câmbio, Selic e IPCA pela API SGS do Banco Central
  2. Lê as planilhas do CEPEA em data/raw/ (download manual, o CEPEA não tem API)
  3. Monta as features
  4. Valida Prophet, Random Forest e XGBoost em walk-forward (12 meses, horizonte 1)
  5. Baixa o relatório WASDE do USDA do mês e gera a previsão com faixa para 1 a 6 meses
     (soja: média do ingênuo com Ridge direto; milho: Ridge com dados do USDA; faixa conformal)
  6. Gera as figuras
  7. Guarda a previsão do próximo mês em reports/historico_previsoes.csv
  8. Escreve reports/atualizacao_mensal.md (com rascunho de post) e atualiza o README

Uso:
  python src/atualizar.py            # atualiza o BCB e roda tudo
  python src/atualizar.py --sem-bcb  # usa os parquets do BCB que já estão no repositório
  python src/atualizar.py --sem-usda # não baixa relatórios novos do WASDE
"""

import argparse
import logging
import os
import re
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
log = logging.getLogger("agro_ml")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
README = ROOT / "README.md"
HISTORICO = REPORT_DIR / "historico_previsoes.csv"

INICIO_HISTORICO = date(2016, 1, 1)
HORIZONTE = 24
N_TEST = 12
COMMODITIES = ["soja", "milho"]
COLHEITA = {"soja": [3, 4, 5], "milho": [2, 3, 7, 8]}
NOME_SERIE = {
    "soja": "Indicador da Soja CEPEA/ESALQ, Paraná",
    "milho": "Indicador do Milho ESALQ/BM&FBovespa",
}
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
         "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]

# Mesmas features e parâmetros do notebook 05
FEATURES = [
    "preco_lag1", "preco_lag2", "preco_lag3",
    "retorno_lag1",
    "cambio_lag1", "cambio_lag3",
    "mes", "trimestre", "colheita",
    "sazonalidade_stl",
]
TARGET = "preco_rs"
MODELOS = {
    "Random Forest": (
        RandomForestRegressor,
        {"n_estimators": 200, "max_depth": 6, "random_state": 42, "n_jobs": -1},
    ),
    "XGBoost": (
        XGBRegressor,
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "random_state": 42, "verbosity": 0},
    ),
}

COR = {"soja": "#2e7d32", "milho": "#c77c02", "rf": "#1565c0", "prophet": "#c2185b"}


# ---------------------------------------------------------------------------
# 1. Banco Central (SGS)
# ---------------------------------------------------------------------------

BCB_SERIES = {"usd_brl": 1, "selic": 21619, "ipca": 433}
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"


def fetch_bcb(serie_code, inicio, fim, max_retries=3, backoff=2.0):
    """Busca uma série do SGS em janelas de até 5 anos (o SGS limita séries diárias a 10 anos por consulta)."""
    partes = []
    janela_ini = inicio
    while janela_ini <= fim:
        janela_fim = min(date(janela_ini.year + 5, janela_ini.month, 1), fim)
        params = {
            "formato": "json",
            "dataInicial": janela_ini.strftime("%d/%m/%Y"),
            "dataFinal": janela_fim.strftime("%d/%m/%Y"),
        }
        for tentativa in range(1, max_retries + 1):
            try:
                resp = requests.get(BCB_URL.format(serie=serie_code), params=params, timeout=60)
                resp.raise_for_status()
                dados = resp.json()
                if dados:
                    partes.append(pd.DataFrame(dados))
                break
            except (requests.exceptions.RequestException, ValueError) as exc:
                log.warning("SGS %s, tentativa %d/%d falhou: %s", serie_code, tentativa, max_retries, exc)
                if tentativa == max_retries:
                    raise
                time.sleep(backoff ** tentativa)
        janela_ini = date(janela_fim.year, janela_fim.month, janela_fim.day)
        if janela_fim == fim:
            break
    if not partes:
        return pd.DataFrame(columns=["data", "valor"])
    df = pd.concat(partes, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.drop_duplicates("data").sort_values("data").reset_index(drop=True)


def atualizar_bcb():
    hoje = date.today()
    for nome, codigo in BCB_SERIES.items():
        caminho = RAW_DIR / f"bcb_{nome}.parquet"
        existente = pd.read_parquet(caminho) if caminho.exists() else pd.DataFrame(columns=["data", nome])

        # Se o histórico salvo começa depois de jan/2016, baixa tudo de novo; senão, só os últimos 90 dias
        if existente.empty or existente["data"].min().to_period("M") > pd.Period(INICIO_HISTORICO, "M"):
            inicio = INICIO_HISTORICO
        else:
            inicio = (existente["data"].max() - pd.Timedelta(days=90)).date()

        try:
            novo = fetch_bcb(codigo, inicio, hoje).rename(columns={"valor": nome})
        except Exception as exc:  # a rodada segue com o que já está salvo
            log.warning("Não consegui atualizar %s pelo Banco Central (%s). Seguindo com os dados salvos.", nome, exc)
            continue
        df = pd.concat([existente, novo], ignore_index=True)
        df = df.drop_duplicates("data", keep="last").sort_values("data").reset_index(drop=True)
        df.to_parquet(caminho, index=False)
        log.info("BCB %s: %d registros | %s a %s", nome, len(df), df["data"].min().date(), df["data"].max().date())

    macro = None
    for nome in BCB_SERIES:
        df = pd.read_parquet(RAW_DIR / f"bcb_{nome}.parquet")
        macro = df if macro is None else macro.merge(df, on="data", how="outer")
    macro.sort_values("data").reset_index(drop=True).to_parquet(RAW_DIR / "bcb_macro.parquet", index=False)


# ---------------------------------------------------------------------------
# 2. CEPEA
# ---------------------------------------------------------------------------

def ler_planilha_cepea(path):
    """
    Lê o .xls que sai do portal do CEPEA (o arquivo vem com um defeito na estrutura interna que o
    xlrd recusa por padrão, por isso o ignore_workbook_corruption) ou um .xlsx salvo pelo Excel.
    """
    if Path(path).suffix.lower() == ".xls":
        import xlrd
        livro = xlrd.open_workbook(str(path), ignore_workbook_corruption=True)
        return pd.read_excel(livro, header=3, engine="xlrd")
    return pd.read_excel(path, header=3)


def parse_cepea(path, commodity):
    """Mesmo layout do notebook 01: cabeçalho na linha 3, data MM/YYYY, preço com vírgula decimal."""
    df = ler_planilha_cepea(path)
    df = df.iloc[:, :3]
    df.columns = ["data", "preco_rs", "preco_usd"]
    df = df.dropna(subset=["data"])
    df["data"] = pd.to_datetime(df["data"].astype(str).str.strip(), format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["data"])
    for col in ["preco_rs", "preco_usd"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["commodity"] = commodity
    df = df.sort_values("data").reset_index(drop=True)
    return df


def carregar_cepea():
    """
    Aceita cepea_<commodity>.xls (do jeito que sai do site) ou .xlsx.
    Se existirem os dois, usa o que tiver o mês mais recente.
    """
    dados = {}
    for commodity in COMMODITIES:
        candidatos = []
        for ext in [".xls", ".xlsx"]:
            caminho = RAW_DIR / f"cepea_{commodity}{ext}"
            if caminho.exists():
                candidatos.append((caminho, parse_cepea(caminho, commodity)))
        if not candidatos:
            raise FileNotFoundError(f"Nenhuma planilha cepea_{commodity}.xls ou .xlsx em {RAW_DIR}")
        caminho, df = max(candidatos, key=lambda c: c[1]["data"].max())
        log.info("CEPEA %s: %s, %d registros | %s a %s", commodity, caminho.name, len(df),
                 df["data"].min().date(), df["data"].max().date())
        df.to_parquet(RAW_DIR / f"cepea_{commodity}.parquet", index=False)
        dados[commodity] = df
    return dados


# ---------------------------------------------------------------------------
# 3. Features (notebook 03)
# ---------------------------------------------------------------------------

def carregar_macro():
    macro = pd.read_parquet(RAW_DIR / "bcb_macro.parquet")
    macro = macro.set_index("data").resample("ME").last().reset_index()
    macro["data"] = macro["data"].dt.to_period("M").dt.to_timestamp()

    ipca = pd.read_parquet(RAW_DIR / "bcb_ipca.parquet").copy()
    ipca["data"] = pd.to_datetime(ipca["data"]).dt.to_period("M").dt.to_timestamp()
    ipca = ipca.sort_values("data").reset_index(drop=True)
    ipca["ipca_acum"] = (1 + ipca["ipca"] / 100).cumprod()
    ipca["ipca_acum"] = ipca["ipca_acum"] / ipca["ipca_acum"].iloc[0] * 100
    return macro, ipca


def build_features(df_cepea, df_macro, df_ipca, commodity):
    df = df_cepea[["data", "preco_rs"]].copy().sort_values("data").reset_index(drop=True)

    for lag in [1, 2, 3]:
        df[f"preco_lag{lag}"] = df["preco_rs"].shift(lag)

    df["retorno_mm"] = df["preco_rs"].pct_change() * 100
    df["retorno_lag1"] = df["retorno_mm"].shift(1)

    df["mes"] = df["data"].dt.month
    df["trimestre"] = df["data"].dt.quarter
    df["colheita"] = df["mes"].isin(COLHEITA[commodity]).astype(int)

    df = df.merge(df_macro[["data", "usd_brl"]], on="data", how="left")
    df["cambio_lag1"] = df["usd_brl"].shift(1)
    df["cambio_lag3"] = df["usd_brl"].shift(3)

    df = df.merge(df_ipca[["data", "ipca_acum"]], on="data", how="left")
    df["preco_real"] = df["preco_rs"] / df["ipca_acum"] * 100

    serie = df.set_index("data")["preco_rs"]
    stl = STL(serie, period=12, robust=True).fit()
    df["sazonalidade_stl"] = stl.seasonal.values
    df["tendencia_stl"] = stl.trend.values

    df = df[df["data"] >= "2016-07-01"].reset_index(drop=True)
    obrigatorias = ["preco_lag1", "preco_lag2", "preco_lag3", "retorno_lag1", "preco_real", "cambio_lag1", "cambio_lag3"]
    df = df.dropna(subset=obrigatorias).reset_index(drop=True)
    df["commodity"] = commodity
    return df


# ---------------------------------------------------------------------------
# 4. Modelos (notebooks 04 e 05)
# ---------------------------------------------------------------------------

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def avaliar(y_true, y_pred):
    return {
        "MAPE": round(mape(y_true, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
    }


def novo_prophet():
    return Prophet(
        seasonality_mode="additive",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )


def rodar_prophet(df):
    df_p = df[["data", "preco_rs"]].rename(columns={"data": "ds", "preco_rs": "y"})
    preds, reais = [], []
    for i in range(N_TEST):
        corte = len(df_p) - N_TEST + i
        m = novo_prophet()
        m.fit(df_p.iloc[:corte])
        futuro = m.make_future_dataframe(periods=1, freq="MS")
        preds.append(m.predict(futuro).iloc[-1]["yhat"])
        reais.append(df_p.iloc[corte]["y"])

    m_final = novo_prophet()
    m_final.fit(df_p)
    fc = m_final.predict(m_final.make_future_dataframe(periods=HORIZONTE, freq="MS"))
    fc = fc[fc["ds"] > df_p["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    return {"metricas": avaliar(reais, preds), "y_true": np.array(reais), "y_pred": np.array(preds), "forecast": fc}


def walk_forward_sklearn(df, modelo_cls, params):
    preds, reais = [], []
    for i in range(N_TEST):
        corte = len(df) - N_TEST + i
        treino, teste = df.iloc[:corte], df.iloc[corte:corte + 1]
        model = modelo_cls(**params)
        model.fit(treino[FEATURES], treino[TARGET])
        preds.append(model.predict(teste[FEATURES])[0])
        reais.append(teste[TARGET].values[0])
    return np.array(reais), np.array(preds)


def recursive_forecast(df, modelo_cls, params, commodity):
    """
    Igual ao notebook 05, com uma correção: a flag de colheita passa a usar o mês previsto
    e o calendário da commodity certa (antes usava a flag do mês anterior e sempre o calendário do milho).
    O câmbio continua congelado no último valor conhecido, como no notebook.
    """
    model = modelo_cls(**params)
    model.fit(df[FEATURES], df[TARGET])

    hist = df.copy()
    previsoes = []
    ultima_data = hist["data"].max()
    for _ in range(HORIZONTE):
        proxima = ultima_data + pd.DateOffset(months=1)
        ultimo = hist.iloc[-1]
        row = {
            "preco_lag1": hist["preco_rs"].iloc[-1],
            "preco_lag2": hist["preco_rs"].iloc[-2],
            "preco_lag3": hist["preco_rs"].iloc[-3],
            "retorno_lag1": (hist["preco_rs"].iloc[-1] / hist["preco_rs"].iloc[-2] - 1) * 100,
            "cambio_lag1": hist["usd_brl"].iloc[-1],
            "cambio_lag3": hist["usd_brl"].iloc[-3],
            "mes": proxima.month,
            "trimestre": proxima.quarter,
            "colheita": int(proxima.month in COLHEITA[commodity]),
            "sazonalidade_stl": ultimo["sazonalidade_stl"],
        }
        pred = model.predict(pd.DataFrame([row])[FEATURES])[0]
        nova = pd.DataFrame([{**row, "data": proxima, "preco_rs": pred, "usd_brl": ultimo["usd_brl"]}])
        hist = pd.concat([hist, nova], ignore_index=True)
        previsoes.append({"ds": proxima, "yhat": pred})
        ultima_data = proxima
    return pd.DataFrame(previsoes)


def rodar_sklearn(df, commodity):
    modelos = {}
    for nome, (cls, params) in MODELOS.items():
        y_true, y_pred = walk_forward_sklearn(df, cls, params)
        modelos[nome] = {"metricas": avaliar(y_true, y_pred), "y_true": y_true, "y_pred": y_pred, "cls": cls, "params": params}
        log.info("%s %s: MAPE %.2f%%", commodity, nome, modelos[nome]["metricas"]["MAPE"])
    melhor = min(modelos, key=lambda n: modelos[n]["metricas"]["MAPE"])
    fc = recursive_forecast(df, modelos[melhor]["cls"], modelos[melhor]["params"], commodity)
    return modelos, melhor, fc


# ---------------------------------------------------------------------------
# 4a. Estimativas de oferta e demanda do USDA (WASDE)
# ---------------------------------------------------------------------------
#
# O USDA publica todo mês (por volta do dia 10) o relatório WASDE e guarda os dados de cada
# relatório como foram publicados na época. Daqui saem, para soja e milho, no ano-safra mais
# recente de cada relatório:
#   - su_mundo / su_eua: estoque final dividido pelo consumo total (mundo e EUA)
#   - rev_*: revisão em relação ao relatório anterior (log da razão) da produção do Brasil,
#     do estoque final mundial e do estoque final dos EUA
# O histórico de 2016 em diante está em data/raw/wasde_features.csv. A cada rodada, o pipeline
# baixa os relatórios que faltam. O relatório do mês T sai antes de o CEPEA do mês T fechar,
# então usar o relatório de T numa previsão feita em T não vaza informação.
# Dados do governo americano, de domínio público.

WASDE_ARQ = RAW_DIR / "wasde_features.csv"
WASDE_URL = "https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-{ano}-{mes:02d}{sufixo}.csv"
WASDE_TITULOS = {"World Soybean Supply and Use": "soja", "World Corn Supply and Use": "milho"}
WASDE_REGIOES = {"World": "W", "United States": "U", "Brazil": "B"}
WASDE_ATRIBUTOS = {"Production": "P", "Ending Stocks": "E", "Domestic Total": "D"}
WASDE_REVISOES = ["rev_br_prod", "rev_mundo_estoque", "rev_eua_estoque"]


def baixar_wasde(ano, mes):
    """Baixa o CSV de um relatório. Tenta primeiro as versões corrigidas (-V3, -V2). Retorna None se não existir."""
    import csv
    import io
    for sufixo in ["-V3", "-V2", ""]:
        url = WASDE_URL.format(ano=ano, mes=mes, sufixo=sufixo)
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "agro_ml (github.com/augustopbueno/agro_ml)"})
        except requests.exceptions.RequestException as exc:
            log.warning("WASDE %d-%02d: falha de rede (%s)", ano, mes, exc)
            return None
        if resp.status_code == 200 and resp.text.lstrip().startswith(("WasdeNumber", '"WasdeNumber"')):
            valores = {}
            for r in csv.DictReader(io.StringIO(resp.text)):
                r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
                c = WASDE_TITULOS.get(r.get("ReportTitle"))
                g = WASDE_REGIOES.get(r.get("Region"))
                a = WASDE_ATRIBUTOS.get(r.get("Attribute"))
                if c and g and a:
                    try:
                        valores[(c, g, a, r["MarketYear"])] = float(r["Value"])
                    except ValueError:
                        pass
            return valores or None
    return None


def features_wasde(atual, anterior):
    """Mesma conta usada para montar o histórico: estoque/uso e revisões no ano-safra mais recente."""
    linha = {}
    for c in ["soja", "milho"]:
        anos = sorted({k[3] for k in atual if k[0] == c and k[1] == "W" and k[2] == "E"})
        L = anos[-1]
        def v(rel, g, a):
            return rel.get((c, g, a, L)) if rel else None
        linha[f"{c}_su_mundo"] = v(atual, "W", "E") / v(atual, "W", "D")
        linha[f"{c}_su_eua"] = v(atual, "U", "E") / v(atual, "U", "D")
        for nome, (g, a) in zip(WASDE_REVISOES, [("B", "P"), ("W", "E"), ("U", "E")]):
            agora, antes = v(atual, g, a), v(anterior, g, a)
            linha[f"{c}_{nome}"] = float(np.log(agora / antes)) if (agora and antes) else 0.0
    return linha


def atualizar_wasde():
    hist = pd.read_csv(WASDE_ARQ, parse_dates=["mes"])
    ultimo = hist["mes"].max()
    hoje = pd.Timestamp(date.today()).to_period("M").to_timestamp()
    if ultimo >= hoje:
        return
    anterior = baixar_wasde(ultimo.year, ultimo.month)
    if anterior is None:
        log.warning("WASDE: não consegui baixar o relatório de %s; as revisões do próximo mês ficam zeradas", ultimo.strftime("%Y-%m"))
    novos = []
    mes = ultimo + pd.DateOffset(months=1)
    while mes <= hoje:
        atual = baixar_wasde(mes.year, mes.month)
        if atual is None:
            log.info("WASDE %s ainda não publicado (ou indisponível)", mes.strftime("%Y-%m"))
        else:
            novos.append({"mes": mes, **features_wasde(atual, anterior)})
            anterior = atual
        mes += pd.DateOffset(months=1)
    if novos:
        hist = pd.concat([hist, pd.DataFrame(novos)], ignore_index=True)
        numericas = hist.columns.drop("mes")
        hist[numericas] = hist[numericas].round(4)
        hist.to_csv(WASDE_ARQ, index=False, date_format="%Y-%m-%d")
        log.info("WASDE: %d relatório(s) novo(s), último %s", len(novos), novos[-1]["mes"].strftime("%Y-%m"))


def carregar_wasde(ate):
    """Série mensal até a data pedida. Mês sem relatório repete o último publicado, com revisão zero."""
    w = pd.read_csv(WASDE_ARQ, parse_dates=["mes"]).set_index("mes").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), max(w.index.max(), ate), freq="MS"))
    rev = [c for c in w.columns if "_rev_" in c]
    w[rev] = w[rev].fillna(0.0)
    return w.ffill()


# ---------------------------------------------------------------------------
# 4b. Previsão com faixa (1 a 6 meses)
# ---------------------------------------------------------------------------
#
# Testamos várias formas de montar a faixa num backtest com 78 origens (jun/2019 a out/2025).
# O que funcionou melhor:
#   - soja: média entre o ingênuo (repete o último preço) e um Ridge direto por horizonte,
#     que prevê o retorno log de t até t+h com informação disponível em t. As variáveis do
#     USDA não melhoraram a soja.
#   - milho: Ridge direto por horizonte com as variáveis do USDA (estoque/uso e revisões),
#     padronizadas. Em 6 meses, o erro caiu de ~18% para ~14% e a faixa de 80% ficou ~35% mais
#     estreita, nos dois subperíodos testados (2022-23 e 2024-25).
#   - faixa: conformal simétrica, isto é, o quantil 80% (ou 95%) do erro absoluto em log das
#     previsões passadas do mesmo horizonte, só usando origens cujo valor real já era conhecido
# O intervalo próprio do Prophet ficou largo e mesmo assim acertou só ~50% das vezes,
# por isso ele não é mais usado para a faixa.

HORIZONTE_FAIXA = 6
NIVEIS_FAIXA = (0.80, 0.95)
ORIGEM_INICIAL_FAIXA = pd.Timestamp("2019-06-01")
INICIO_AVALIACAO_FAIXA = pd.Timestamp("2022-01-01")
MIN_RESIDUOS = 12
METODO_FAIXA = {"soja": "media", "milho": "usda"}
NOME_METODO_FAIXA = {"media": "Média ingênuo + Ridge", "usda": "Ridge com dados do USDA"}


def features_faixa(p, cambio, wasde=None, commodity=None):
    lp, lc = np.log(p), np.log(cambio)
    X = pd.DataFrame(index=p.index)
    for k in [1, 3, 6, 12]:
        X[f"r{k}"] = lp - lp.shift(k)
    for k in [1, 3]:
        X[f"c{k}"] = lc - lc.shift(k)
    X["desvio12"] = lp - np.log(p.rolling(12).mean())
    X["mes_sin"] = np.sin(2 * np.pi * p.index.month / 12)
    X["mes_cos"] = np.cos(2 * np.pi * p.index.month / 12)
    if wasde is not None:
        w = wasde.reindex(p.index)
        for nome in ["su_mundo", "su_eua"]:
            X[f"w_{nome}"] = w[f"{commodity}_{nome}"]
        for nome in WASDE_REVISOES:
            col = w[f"{commodity}_{nome}"]
            X[f"w_{nome}_1"] = col
            X[f"w_{nome}_3"] = col.rolling(3).sum()
    return X


def prever_ponto(p, X, T, h, metodo):
    """Previsão de t+h feita em T. Treina só com pares cujo alvo já era conhecido em T."""
    lp = np.log(p)
    alvo = lp.shift(-h) - lp
    idx = X.index[X.index <= T - pd.DateOffset(months=h)]
    treino = pd.concat([X.loc[idx], alvo.loc[idx].rename("y")], axis=1).dropna()
    if metodo == "usda":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        modelo = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(treino.drop(columns="y"), treino["y"])
        return float(np.exp(lp.loc[T] + modelo.predict(X.loc[[T]])[0]))
    ridge = Ridge(alpha=1.0).fit(treino.drop(columns="y"), treino["y"])
    prev_ridge = float(np.exp(lp.loc[T] + ridge.predict(X.loc[[T]])[0]))
    return (float(p.loc[T]) + prev_ridge) / 2


def rodar_faixa(df, commodity, wasde):
    p = df.set_index("data")["preco_rs"]
    metodo = METODO_FAIXA[commodity]
    X = features_faixa(p, df.set_index("data")["usd_brl"],
                       wasde if metodo == "usda" else None, commodity)
    ultimo = p.index.max()

    # Backtest: todas as origens desde 2019-06 até o mês anterior ao último
    linhas = []
    for T in p.index[(p.index >= ORIGEM_INICIAL_FAIXA) & (p.index < ultimo)]:
        for h in range(1, HORIZONTE_FAIXA + 1):
            alvo = T + pd.DateOffset(months=h)
            if alvo > ultimo:
                break
            linhas.append({"origem": T, "h": h, "prev": prever_ponto(p, X, T, h, metodo), "real": float(p.loc[alvo]),
                           "ingenuo": float(p.loc[T])})
    bt = pd.DataFrame(linhas)
    bt["err"] = np.abs(np.log(bt["real"] / bt["prev"]))

    # Acerto histórico: em cada origem desde 2022, monta a faixa só com o que se sabia na época
    acerto = {}
    for h in range(1, HORIZONTE_FAIXA + 1):
        g = bt[bt["h"] == h]
        dentro = {n: [] for n in NIVEIS_FAIXA}
        for _, r in g[g["origem"] >= INICIO_AVALIACAO_FAIXA].iterrows():
            pool = g[g["origem"] <= r["origem"] - pd.DateOffset(months=h)]["err"]
            if len(pool) < MIN_RESIDUOS:
                continue
            for n in NIVEIS_FAIXA:
                dentro[n].append(r["err"] <= np.quantile(pool, n))
        acerto[h] = {n: (np.mean(v) * 100 if v else np.nan) for n, v in dentro.items()}
        acerto[h]["n"] = len(dentro[NIVEIS_FAIXA[0]])

    # Previsão atual, com a faixa calculada sobre todos os erros já conhecidos
    fc = []
    for h in range(1, HORIZONTE_FAIXA + 1):
        prev = prever_ponto(p, X, ultimo, h, metodo)
        pool = bt[bt["h"] == h]["err"]
        linha = {"ds": ultimo + pd.DateOffset(months=h), "h": h, "prev": prev}
        for n in NIVEIS_FAIXA:
            q = np.quantile(pool, n)
            linha[f"lo{int(n * 100)}"] = prev * np.exp(-q)
            linha[f"hi{int(n * 100)}"] = prev * np.exp(q)
        fc.append(linha)

    # Erro de 1 mês nos mesmos 12 meses da tabela de validação, para comparar com os outros modelos
    ult12 = bt[(bt["h"] == 1)].tail(N_TEST)
    return {
        "forecast": pd.DataFrame(fc),
        "acerto": acerto,
        "backtest": bt,
        "metodo": NOME_METODO_FAIXA[metodo],
        "mape_media": round(mape(ult12["real"], ult12["prev"]), 2),
        "mape_ingenuo": round(mape(ult12["real"], ult12["ingenuo"]), 2),
    }


# ---------------------------------------------------------------------------
# 5. Figuras
# ---------------------------------------------------------------------------

def figuras(datasets, resultados):
    sns.set_theme(style="whitegrid", palette="muted")
    cores_hist = {"soja": "#2e7d32", "milho": "#f9a825"}

    # Prophet: histórico + 24 meses
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, commodity in zip(axes, COMMODITIES):
        df, r = datasets[commodity], resultados[commodity]["prophet"]
        fc, m = r["forecast"], r["metricas"]
        ax.plot(df["data"], df["preco_rs"], color=cores_hist[commodity], linewidth=1.8, label="Histórico")
        ax.plot(fc["ds"], fc["yhat"], color="#1565c0", linewidth=1.8, linestyle="--", label="Previsão")
        ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.2, color="#1565c0", label="IC 95%")
        ax.axvline(df["data"].max(), color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{commodity.capitalize()}, Prophet | MAPE: {m['MAPE']}% | RMSE: R${m['RMSE']} | MAE: R${m['MAE']}")
        ax.set_ylabel("R$/saca")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prophet_previsao.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Prophet: período de teste
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, commodity in zip(axes, COMMODITIES):
        r = resultados[commodity]["prophet"]
        idx = range(len(r["y_true"]))
        ax.plot(idx, r["y_true"], color=cores_hist[commodity], linewidth=1.8, marker="o", markersize=4, label="Real")
        ax.plot(idx, r["y_pred"], color="#1565c0", linewidth=1.5, marker="s", markersize=4, linestyle="--", label="Previsto")
        ax.set_title(f"{commodity.capitalize()}, real vs previsto (teste 12 meses)")
        ax.set_ylabel("R$/saca")
        ax.set_xlabel("Mês (t)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prophet_teste.png", dpi=150, bbox_inches="tight")
    plt.close()

    # sklearn: período de teste
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, commodity in enumerate(COMMODITIES):
        for col, nome in enumerate(MODELOS):
            ax = axes[row][col]
            r = resultados[commodity]["sklearn"][nome]
            n = len(r["y_true"])
            ax.plot(range(n), r["y_true"], color=cores_hist[commodity], linewidth=1.8, marker="o", markersize=4, label="Real")
            ax.plot(range(n), r["y_pred"], color="#1565c0", linewidth=1.5, marker="s", markersize=4, linestyle="--", label="Previsto")
            ax.set_title(f"{commodity.capitalize()}, {nome} | MAPE: {r['metricas']['MAPE']}%")
            ax.set_ylabel("R$/saca")
            ax.set_xlabel("Mês (t)")
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sklearn_teste.png", dpi=150, bbox_inches="tight")
    plt.close()

    # sklearn: importância das features (Random Forest no histórico completo)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, commodity in zip(axes, COMMODITIES):
        df = datasets[commodity]
        cls, params = MODELOS["Random Forest"]
        model = cls(**params).fit(df[FEATURES], df[TARGET])
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
        imp.plot(kind="barh", ax=ax, color=cores_hist[commodity], alpha=0.85)
        ax.set_title(f"{commodity.capitalize()}, importância das features (RF)")
        ax.set_xlabel("Importância")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sklearn_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # sklearn: previsão 24 meses com o melhor modelo
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, commodity in zip(axes, COMMODITIES):
        df, r = datasets[commodity], resultados[commodity]
        melhor, fc = r["melhor"], r["sklearn_forecast"]
        ax.plot(df["data"], df[TARGET], color=cores_hist[commodity], linewidth=1.8, label="Histórico")
        ax.plot(fc["ds"], fc["yhat"], color="#1565c0", linewidth=1.8, linestyle="--", label=f"Previsão ({melhor})")
        ax.axvline(df["data"].max(), color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{commodity.capitalize()}, {melhor} | MAPE teste: {r['sklearn'][melhor]['metricas']['MAPE']}%")
        ax.set_ylabel("R$/saca")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sklearn_previsao.png", dpi=150, bbox_inches="tight")
    plt.close()

    figura_post(datasets, resultados)


def figura_post(datasets, resultados):
    """
    Figura para o post: preço real dos últimos 36 meses e a previsão dos próximos 6 meses
    com a faixa de 80% (a de 95% fica no relatório).
    """
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, commodity in zip(axes, COMMODITIES):
        df, fx = datasets[commodity], resultados[commodity]["faixa"]
        fc, acerto = fx["forecast"], fx["acerto"]
        hist = df[df["data"] > df["data"].max() - pd.DateOffset(months=36)]
        ultimo_x, ultimo_y = hist["data"].iloc[-1], hist["preco_rs"].iloc[-1]
        xs = [ultimo_x, *fc["ds"]]

        ax.fill_between(xs, [ultimo_y, *fc["lo80"]], [ultimo_y, *fc["hi80"]], color=COR["rf"], alpha=0.20,
                        linewidth=0, label="Faixa de 80%")
        ax.plot(hist["data"], hist["preco_rs"], color=COR[commodity], linewidth=2, label="CEPEA (real)", zorder=3)
        ax.plot(xs, [ultimo_y, *fc["prev"]], color=COR["rf"], linewidth=1.8, linestyle="--", label="Previsão", zorder=3)

        p1 = fc.iloc[0]
        ax.scatter([p1["ds"]], [p1["prev"]], s=40, color=COR["rf"], edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(f"{mes_ano(p1['ds'])}: R$ {formatar_br(p1['prev'], 0)}\n80%: R$ {formatar_br(p1['lo80'], 0)} a {formatar_br(p1['hi80'], 0)}",
                    (p1["ds"], p1["hi80"]), textcoords="offset points", xytext=(-6, 6), ha="right", va="bottom",
                    fontsize=9.5, color="#222222",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        ax.set_title(f"{commodity.capitalize()} (R$/saca 60 kg)", loc="left", fontsize=13, color="#222222", pad=24)
        ax.text(0, 1.02, f"Acerto da faixa de 80% (1 mês) desde 2022: {acerto[1][0.80]:.0f}%",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="#555555")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for lado in ["top", "right", "left"]:
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(colors="#555555", length=0)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
        ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)

    ref = datasets["soja"]["data"].max()
    fig.suptitle(f"Previsão de preço para os próximos 6 meses, com o CEPEA até {mes_ano(ref)}",
                 x=0.01, ha="left", fontsize=14, color="#222222")
    fig.text(0.01, 0.01, "Fonte: CEPEA/ESALQ, Banco Central e USDA (WASDE). Código: github.com/augustopbueno/agro_ml",
             fontsize=8.5, color="#777777")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(FIG_DIR / "post_linkedin.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Histórico de previsões, relatório e README
# ---------------------------------------------------------------------------

def formatar_br(valor, casas=2):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def mes_ano(ts):
    return f"{MESES[ts.month - 1]}/{ts.year}"


def atualizar_historico(datasets, resultados):
    """Guarda a previsão do próximo mês desta rodada. Uma linha por (referência, commodity, tipo)."""
    linhas = []
    for commodity in COMMODITIES:
        ref = datasets[commodity]["data"].max()
        r = resultados[commodity]
        prox_f = r["faixa"]["forecast"].iloc[0]
        prox_s = r["sklearn_forecast"].iloc[0]
        prox_p = r["prophet"]["forecast"].iloc[0]
        linhas.append({"data_referencia": ref, "commodity": commodity, "tipo": "faixa", "modelo": r["faixa"]["metodo"],
                       "mes_previsto": prox_f["ds"], "previsao": round(float(prox_f["prev"]), 2),
                       "lo80": round(float(prox_f["lo80"]), 2), "hi80": round(float(prox_f["hi80"]), 2),
                       "lo95": round(float(prox_f["lo95"]), 2), "hi95": round(float(prox_f["hi95"]), 2)})
        linhas.append({"data_referencia": ref, "commodity": commodity, "tipo": "sklearn", "modelo": r["melhor"],
                       "mes_previsto": prox_s["ds"], "previsao": round(float(prox_s["yhat"]), 2)})
        linhas.append({"data_referencia": ref, "commodity": commodity, "tipo": "prophet", "modelo": "Prophet",
                       "mes_previsto": prox_p["ds"], "previsao": round(float(prox_p["yhat"]), 2)})
    novo = pd.DataFrame(linhas)
    if HISTORICO.exists():
        antigo = pd.read_csv(HISTORICO, parse_dates=["data_referencia", "mes_previsto"])
        novo = pd.concat([antigo, novo], ignore_index=True)
    novo = novo.drop_duplicates(["data_referencia", "commodity", "tipo"], keep="last")
    novo = novo.sort_values(["data_referencia", "commodity", "tipo"]).reset_index(drop=True)
    novo.to_csv(HISTORICO, index=False, date_format="%Y-%m-%d")
    return novo


def erro_mes_anterior(historico, datasets, commodity, tipo="faixa"):
    """Compara a previsão feita na rodada anterior (e a faixa, se houver) com o valor real que o CEPEA publicou depois."""
    real = datasets[commodity].set_index("data")["preco_rs"]
    h = historico[(historico["commodity"] == commodity) & (historico["tipo"] == tipo)]
    h = h[h["mes_previsto"] == datasets[commodity]["data"].max()]
    if h.empty:
        return None
    linha = h.iloc[-1]
    valor_real = float(real.loc[linha["mes_previsto"]])
    saida = {"mes": linha["mes_previsto"], "modelo": linha["modelo"], "previsto": float(linha["previsao"]),
             "real": valor_real, "erro_pct": abs(linha["previsao"] - valor_real) / valor_real * 100}
    if "lo80" in linha and pd.notna(linha.get("lo80")):
        saida.update({"lo80": float(linha["lo80"]), "hi80": float(linha["hi80"]),
                      "dentro80": linha["lo80"] <= valor_real <= linha["hi80"]})
    return saida


def tabela_metricas(resultados):
    linhas = ["| Modelo | MAPE Soja | MAPE Milho |", "|---|---|---|"]
    def valor(r, nome):
        if nome == "Prophet":
            return r["prophet"]["metricas"]["MAPE"]
        if nome == "Ingênuo (repete o último preço)":
            return r["faixa"]["mape_ingenuo"]
        if nome == "Modelo da faixa (soja: média ingênuo + Ridge; milho: Ridge + USDA)":
            return r["faixa"]["mape_media"]
        return r["sklearn"][nome]["metricas"]["MAPE"]
    for nome in ["Ingênuo (repete o último preço)", "Prophet", *MODELOS, "Modelo da faixa (soja: média ingênuo + Ridge; milho: Ridge + USDA)"]:
        v = [valor(resultados[c], nome) for c in COMMODITIES]
        linhas.append(f"| {nome} | {v[0]:.2f}% | {v[1]:.2f}% |")
    return "\n".join(linhas)


def tabela_faixa(resultados, horizontes):
    linhas = ["| Mês | Soja | Faixa 80% | Milho | Faixa 80% |", "|---|---|---|---|---|"]
    fs, fm = resultados["soja"]["faixa"]["forecast"], resultados["milho"]["faixa"]["forecast"]
    for h in horizontes:
        a, b = fs[fs["h"] == h].iloc[0], fm[fm["h"] == h].iloc[0]
        linhas.append(f"| {mes_ano(a['ds'])} | R$ {formatar_br(a['prev'])} | R$ {formatar_br(a['lo80'])} a {formatar_br(a['hi80'])} "
                      f"| R$ {formatar_br(b['prev'])} | R$ {formatar_br(b['lo80'])} a {formatar_br(b['hi80'])} |")
    return "\n".join(linhas)


def tabela_acerto(resultados):
    linhas = ["| Horizonte | Soja 80% | Soja 95% | Milho 80% | Milho 95% |", "|---|---|---|---|---|"]
    for h in range(1, HORIZONTE_FAIXA + 1):
        a, b = resultados["soja"]["faixa"]["acerto"][h], resultados["milho"]["faixa"]["acerto"][h]
        linhas.append(f"| {h} {'mês' if h == 1 else 'meses'} | {a[0.80]:.0f}% | {a[0.95]:.0f}% | {b[0.80]:.0f}% | {b[0.95]:.0f}% |")
    return "\n".join(linhas)


def montar_textos(datasets, resultados, historico):
    ref = datasets["soja"]["data"].max()
    prox = ref + pd.DateOffset(months=1)
    inicio = datasets["soja"]["data"].min()
    bloco = {}
    for commodity in COMMODITIES:
        fx = resultados[commodity]["faixa"]
        bloco[commodity] = {
            "ultimo": float(datasets[commodity]["preco_rs"].iloc[-1]),
            "f": fx["forecast"].set_index("h"),
            "acerto": fx["acerto"],
            "erro": erro_mes_anterior(historico, datasets, commodity),
        }
    s, m = bloco["soja"], bloco["milho"]

    # Relatório
    rel = [
        f"# Atualização mensal: {mes_ano(ref)}",
        "",
        f"Gerado em {date.today().strftime('%d/%m/%Y')} pelo workflow `atualizacao_mensal`.",
        "",
        "## Dados",
        "",
        f"- Período usado nos modelos: {mes_ano(inicio)} a {mes_ano(ref)} ({len(datasets['soja'])} meses)",
    ]
    for commodity in COMMODITIES:
        rel.append(f"- {commodity.capitalize()}: {NOME_SERIE[commodity]}, último valor R$ {formatar_br(bloco[commodity]['ultimo'])}/saca")
    rel += [
        "",
        "## Previsão com faixa (próximos 6 meses)",
        "",
        "Soja: média entre o ingênuo (repete o último preço) e um Ridge direto por horizonte. "
        "Milho: Ridge direto por horizonte com as estimativas de estoque e as revisões do relatório WASDE do USDA. "
        "Faixa: quantil do erro das previsões passadas do mesmo horizonte (conformal simétrica).",
        "",
        tabela_faixa(resultados, range(1, HORIZONTE_FAIXA + 1)),
        "",
        "### Quanto a faixa acertou no passado",
        "",
        "Percentual de vezes em que o preço real caiu dentro da faixa, em previsões feitas desde jan/2022 "
        "só com a informação disponível em cada data.",
        "",
        tabela_acerto(resultados),
        "",
        "## Validação walk-forward (últimos 12 meses, horizonte de 1 mês)",
        "",
        tabela_metricas(resultados),
        "",
        "## Como foi o mês anterior",
        "",
    ]
    teve_erro = False
    for commodity in COMMODITIES:
        e = bloco[commodity]["erro"]
        if e:
            teve_erro = True
            faixa_txt = ""
            if "lo80" in e:
                faixa_txt = (f", faixa de 80% R$ {formatar_br(e['lo80'])} a {formatar_br(e['hi80'])} "
                             f"({'dentro' if e['dentro80'] else 'fora'})")
            rel.append(f"- {commodity.capitalize()} em {mes_ano(e['mes'])}: previsto R$ {formatar_br(e['previsto'])}{faixa_txt}, "
                       f"real R$ {formatar_br(e['real'])}, erro de {formatar_br(e['erro_pct'])}%")
    if not teve_erro:
        rel.append("- Ainda não há previsão de rodada anterior para comparar. A comparação aparece a partir da próxima atualização.")

    # Rascunho do post
    s1, m1, s3, m3 = s["f"].loc[1], m["f"].loc[1], s["f"].loc[3], m["f"].loc[3]
    post = [f"Atualizei meu modelo de previsão de soja e milho com os dados do CEPEA de {mes_ano(ref)}.", ""]
    if teve_erro:
        partes = []
        for commodity in COMMODITIES:
            e = bloco[commodity]["erro"]
            if e:
                artigo = "na" if commodity == "soja" else "no"
                txt = f"{artigo} {commodity} a previsão era R$ {formatar_br(e['previsto'])} e o CEPEA fechou em R$ {formatar_br(e['real'])}"
                if "dentro80" in e:
                    txt += ", dentro da faixa" if e["dentro80"] else ", fora da faixa"
                partes.append(txt)
        post += [f"Como ele foi no mês passado: {'; '.join(partes)}.", ""]
    post += [
        f"Para {mes_ano(prox)}: soja em R$ {formatar_br(s1['prev'])}, com 80% de chance de ficar entre "
        f"R$ {formatar_br(s1['lo80'])} e R$ {formatar_br(s1['hi80'])}. Milho em R$ {formatar_br(m1['prev'])}, "
        f"entre R$ {formatar_br(m1['lo80'])} e R$ {formatar_br(m1['hi80'])}.",
        "",
        f"Para {mes_ano(s3['ds'])} a faixa já abre: soja entre R$ {formatar_br(s3['lo80'], 0)} e R$ {formatar_br(s3['hi80'], 0)}, "
        f"milho entre R$ {formatar_br(m3['lo80'], 0)} e R$ {formatar_br(m3['hi80'], 0)}. "
        "A incerteza cresce com o prazo, e acima de 6 meses a faixa fica larga demais para ser útil, então não publico.",
        "",
        f"Testei a faixa no passado: desde 2022, a de 80% para o mês seguinte pegou o preço real em "
        f"{s['acerto'][1][0.80]:.0f}% das vezes na soja e {m['acerto'][1][0.80]:.0f}% no milho. "
        "O modelo usa preço passado e câmbio e, no milho, as estimativas de estoque do USDA, "
        "que ajudaram a estreitar a faixa nos prazos maiores. Clima e custo de fertilizante ainda ficam de fora.",
        "",
        "A atualização roda sozinha todo mês pelo GitHub Actions. Código e dados: github.com/augustopbueno/agro_ml",
        "",
        "#dados #machinelearning #agronegocio #python",
    ]
    rel += [
        "",
        "## Rascunho de post para o LinkedIn",
        "",
        "Revise antes de publicar. Imagem sugerida: `reports/figures/post_linkedin.png`",
        "",
        "```text",
        *post,
        "```",
        "",
    ]

    # Bloco do README
    readme_resultados = [
        f"Atualizado com o CEPEA de {mes_ano(ref)}.",
        "",
        "**Previsão com faixa de 80%**",
        "",
        tabela_faixa(resultados, [1, 3, 6]),
        "",
        f"Desde 2022, a faixa de 80% do mês seguinte pegou o preço real em {s['acerto'][1][0.80]:.0f}% das vezes "
        f"na soja e {m['acerto'][1][0.80]:.0f}% no milho.",
        "",
        "**Validação walk-forward, últimos 12 meses, horizonte de 1 mês**",
        "",
        tabela_metricas(resultados),
        "",
        "Detalhes, faixa de 95% e histórico em [reports/atualizacao_mensal.md](reports/atualizacao_mensal.md).",
    ]
    readme_periodo = f"Período usado nos modelos: {mes_ano(inicio)} a {mes_ano(ref)}."
    return "\n".join(rel), "\n".join(readme_resultados), readme_periodo


def substituir_bloco(texto, nome, conteudo):
    padrao = re.compile(rf"(<!-- {nome}:INICIO -->\n).*?(\n<!-- {nome}:FIM -->)", re.S)
    if not padrao.search(texto):
        raise RuntimeError(f"Marcadores {nome} não encontrados no README")
    return padrao.sub(lambda mt: mt.group(1) + conteudo + mt.group(2), texto)


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def cepea_em_dia(datasets):
    """O CEPEA do mês anterior deveria estar disponível. Retorna False se a planilha está atrasada."""
    esperado = (pd.Timestamp(date.today()).to_period("M") - 1).to_timestamp()
    ultimo_cepea = min(pd.read_parquet(RAW_DIR / f"cepea_{c}.parquet")["data"].max() for c in COMMODITIES)
    return ultimo_cepea >= esperado, ultimo_cepea, esperado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-bcb", action="store_true", help="não consulta a API do Banco Central")
    parser.add_argument("--sem-usda", action="store_true", help="não baixa relatórios novos do WASDE")
    args = parser.parse_args()

    for pasta in [PROC_DIR, FIG_DIR, REPORT_DIR]:
        pasta.mkdir(parents=True, exist_ok=True)

    if not args.sem_bcb:
        atualizar_bcb()
    if not args.sem_usda:
        try:
            atualizar_wasde()
        except Exception as exc:  # a rodada segue com o histórico salvo
            log.warning("Não consegui atualizar o WASDE (%s). Seguindo com os dados salvos.", exc)

    cepea = carregar_cepea()
    macro, ipca = carregar_macro()
    wasde = carregar_wasde(max(df["data"].max() for df in cepea.values()))

    datasets, resultados = {}, {}
    for commodity in COMMODITIES:
        df = build_features(cepea[commodity], macro, ipca, commodity)
        df.to_parquet(PROC_DIR / f"features_{commodity}.parquet", index=False)
        datasets[commodity] = df
        log.info("Features %s: %d meses | %s a %s", commodity, len(df), df["data"].min().date(), df["data"].max().date())

        prophet_res = rodar_prophet(df)
        log.info("%s Prophet: MAPE %.2f%%", commodity, prophet_res["metricas"]["MAPE"])
        sk, melhor, sk_fc = rodar_sklearn(df, commodity)
        faixa = rodar_faixa(df, commodity, wasde)
        log.info("%s faixa: próximo mês R$ %.2f (80%%: %.2f a %.2f) | acerto histórico 80%%: %.0f%%", commodity,
                 faixa["forecast"].iloc[0]["prev"], faixa["forecast"].iloc[0]["lo80"], faixa["forecast"].iloc[0]["hi80"],
                 faixa["acerto"][1][0.80])
        resultados[commodity] = {"prophet": prophet_res, "sklearn": sk, "melhor": melhor, "sklearn_forecast": sk_fc,
                                 "faixa": faixa}

    figuras(datasets, resultados)
    historico = atualizar_historico(datasets, resultados)
    relatorio, readme_res, readme_periodo = montar_textos(datasets, resultados, historico)
    (REPORT_DIR / "atualizacao_mensal.md").write_text(relatorio, encoding="utf-8")

    texto = README.read_text(encoding="utf-8")
    texto = substituir_bloco(texto, "RESULTADOS", readme_res)
    texto = substituir_bloco(texto, "PERIODO", readme_periodo)
    README.write_text(texto, encoding="utf-8")

    em_dia, ultimo, esperado = cepea_em_dia(datasets)
    log.info("CEPEA em dia: %s (último %s, esperado %s)", em_dia, ultimo.date(), esperado.date())
    saida = os.environ.get("GITHUB_OUTPUT")
    if saida:
        with open(saida, "a", encoding="utf-8") as f:
            f.write(f"cepea_em_dia={'true' if em_dia else 'false'}\n")
            f.write(f"mes_esperado={MESES[esperado.month - 1]}/{esperado.year}\n")
            f.write(f"referencia={datasets['soja']['data'].max().strftime('%Y-%m')}\n")


if __name__ == "__main__":
    main()
