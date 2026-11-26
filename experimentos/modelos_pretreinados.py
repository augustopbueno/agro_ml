"""
Compara modelos pré-treinados de séries temporais (Chronos-2, TimesFM 2.5 e TimesFM 3)
com os modelos do agro_ml, no mesmo backtest usado para escolher a faixa:
origens de jun/2019 a out/2025, horizontes de 1 a 6 meses, avaliação desde jan/2022.

Cada modelo recebe, em cada origem, só o histórico até aquela data (sem vazamento).
A faixa de 80% é medida de dois jeitos:
  - conformal simétrica sobre os erros passados (igual à do pipeline)
  - faixa nativa do modelo (quantis 10% e 90%), quando ele fornece

Uso:
  python modelos_pretreinados.py            # roda os modelos de verdade (precisa do Hugging Face)
  python modelos_pretreinados.py --dummy    # testa o código com um modelo falso (repete o último preço)
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

H = 6
ORIG_INI, EVAL_INI, ORIG_FIM = pd.Timestamp("2019-06-01"), pd.Timestamp("2022-01-01"), pd.Timestamp("2025-10-01")
DADOS = os.environ.get("AGRO_DADOS", "data/processed/features_{c}.parquet")
WASDE = os.environ.get("AGRO_WASDE", "data/raw/wasde_features.csv")


def carregar(c):
    df = pd.read_parquet(DADOS.format(c=c)).set_index("data")[["preco_rs", "usd_brl"]]
    w = pd.read_csv(WASDE, parse_dates=["mes"]).set_index("mes")
    w = w.reindex(pd.date_range(w.index.min(), max(w.index.max(), df.index.max()), freq="MS"))
    rev = [x for x in w.columns if "_rev_" in x]
    w[rev] = w[rev].fillna(0.0)
    w = w.ffill()
    df["su_mundo"] = w[f"{c}_su_mundo"].reindex(df.index).values
    df["su_eua"] = w[f"{c}_su_eua"].reindex(df.index).values
    return df


def origens(df):
    return [t for t in df.index if ORIG_INI <= t <= ORIG_FIM]


# ---------------------------------------------------------------------------
# Adaptadores: cada um recebe a lista de contextos e devolve (mediana, q10, q90) com shape (n, H)
# ---------------------------------------------------------------------------

def modelo_dummy(contextos, covs=None):
    ultimo = np.array([c[-1] for c in contextos])
    med = np.repeat(ultimo[:, None], H, axis=1)
    return med, med * 0.93, med * 1.07


def carregar_chronos2():
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    def prever(contextos, covs=None):
        if covs is None:
            entradas = [np.asarray(c, dtype=np.float32) for c in contextos]
        else:
            entradas = [{"target": np.asarray(c, dtype=np.float32),
                         "past_covariates": {k: np.asarray(v, dtype=np.float32) for k, v in cv.items()}}
                        for c, cv in zip(contextos, covs)]
        qs, _ = pipe.predict_quantiles(entradas, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
        arr = np.stack([q.squeeze(0).numpy() if q.dim() == 3 else q.numpy() for q in qs])  # (n, H, 3)
        return arr[:, :, 1], arr[:, :, 0], arr[:, :, 2]
    return prever


def carregar_timesfm25():
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=256, max_horizon=H, normalize_inputs=True,
                                     use_continuous_quantile_head=True, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))

    def prever(contextos, covs=None):
        ponto, quant = m.forecast(horizon=H, inputs=[np.asarray(c, dtype=np.float32) for c in contextos])
        # quant: (n, H, 10) = média, q10, q20, ..., q90
        return ponto[:, :H], quant[:, :H, 1], quant[:, :H, 9]
    return prever


def carregar_timesfm3():
    import timesfm
    f = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")

    def prever(contextos, covs=None):
        saidas = list(f.predict_batch(contexts=[np.asarray(c, dtype=np.float32) for c in contextos],
                                      horizon=H, return_quantiles=True))
        med = np.stack([np.asarray(o.forecast)[:H] for o in saidas])
        q10 = q90 = None
        q = saidas[0].quantiles
        if q is not None:
            q = np.asarray(q)
            # layout esperado: (H, nq) com quantis de 0.1 a 0.9; se for diferente, fica sem faixa nativa
            if q.ndim == 2 and q.shape[0] >= H and q.shape[1] >= 9:
                Q = np.stack([np.asarray(o.quantiles)[:H] for o in saidas])
                q10, q90 = Q[:, :, -9], Q[:, :, -1]
        return med, q10, q90
    return prever


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def rodar(nome, prever, com_covs=False):
    linhas = []
    for c in ["soja", "milho"]:
        df = carregar(c)
        ts = origens(df)
        contextos = [df.loc[:T, "preco_rs"].values for T in ts]
        covs = None
        if com_covs:
            covs = [{"log_cambio": np.log(df.loc[:T, "usd_brl"].values),
                     "su_mundo": df.loc[:T, "su_mundo"].values,
                     "su_eua": df.loc[:T, "su_eua"].values} for T in ts]
        t0 = time.time()
        med, q10, q90 = prever(contextos, covs)
        print(f"  {nome} {c}: {len(ts)} origens em {time.time() - t0:.0f}s", file=sys.stderr)
        for i, T in enumerate(ts):
            for h in range(1, H + 1):
                alvo = T + pd.DateOffset(months=h)
                if alvo not in df.index:
                    break
                linhas.append({"commodity": c, "origem": T, "h": h, "metodo": nome,
                               "prev": float(med[i, h - 1]), "real": float(df.loc[alvo, "preco_rs"]),
                               "q10": float(q10[i, h - 1]) if q10 is not None else np.nan,
                               "q90": float(q90[i, h - 1]) if q90 is not None else np.nan})
    return pd.DataFrame(linhas)


def avaliar(bt):
    bt = bt.copy()
    bt["err"] = np.abs(np.log(bt["real"] / bt["prev"]))
    res = []
    for (c, met, h), g in bt.groupby(["commodity", "metodo", "h"]):
        g = g.sort_values("origem")
        dentro, larg, score, erro, nat_dentro, nat_larg = [], [], [], [], [], []
        for _, r in g[g["origem"] >= EVAL_INI].iterrows():
            erro.append(abs(r["prev"] - r["real"]) / r["real"] * 100)
            if pd.notna(r.get("q10", np.nan)):
                nat_dentro.append(r["q10"] <= r["real"] <= r["q90"])
                nat_larg.append((r["q90"] - r["q10"]) / r["real"] * 100)
            pool = g[g["origem"] <= r["origem"] - pd.DateOffset(months=h)]["err"]
            if len(pool) < 12:
                continue
            q = np.quantile(pool, 0.8)
            lo, hi = r["prev"] * np.exp(-q), r["prev"] * np.exp(q)
            dentro.append(lo <= r["real"] <= hi)
            larg.append((hi - lo) / r["real"] * 100)
            score.append(((hi - lo) + 10 * max(lo - r["real"], 0) + 10 * max(r["real"] - hi, 0)) / r["real"] * 100)
        res.append({"commodity": c, "metodo": met, "h": h, "erro": np.mean(erro),
                    "acerto80": np.mean(dentro) * 100, "largura80": np.mean(larg), "score": np.mean(score),
                    "acerto_nativo": np.mean(nat_dentro) * 100 if nat_dentro else np.nan,
                    "largura_nativa": np.mean(nat_larg) if nat_larg else np.nan, "n": len(erro)})
    return pd.DataFrame(res)


def referencias():
    """Modelos do agro_ml no mesmo backtest (gerados antes por faixa_wasde.py)."""
    bt = pd.read_parquet("/home/claude/experimentos/backtest_wasde.parquet")
    bt = bt[bt["h"] <= H]
    keep = {"ingenuo": "ingenuo (repete o último preço)",
            "media_ing_ridge": "agro_ml soja (média ingênuo + Ridge)",
            "ridge_wasde_esc_puro": "agro_ml milho (Ridge + USDA)"}
    bt = bt[bt["metodo"].isin(keep)].copy()
    # cada commodity compara com o modelo que o pipeline usa para ela
    bt = bt[~((bt["commodity"] == "soja") & (bt["metodo"] == "ridge_wasde_esc_puro"))]
    bt = bt[~((bt["commodity"] == "milho") & (bt["metodo"] == "media_ing_ridge"))]
    bt["metodo"] = bt["metodo"].map(keep)
    return bt[["commodity", "origem", "h", "metodo", "prev", "real"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--so-previsoes", action="store_true",
                    help="só gera as previsões (sem comparar com os modelos do agro_ml)")
    args = ap.parse_args()

    partes = [] if args.so_previsoes else [referencias()]
    if args.dummy:
        partes.append(rodar("dummy (teste do código)", modelo_dummy))
    else:
        for nome, carregador, covs in [("Chronos-2", carregar_chronos2, False),
                                        ("Chronos-2 + câmbio e USDA", carregar_chronos2, True),
                                        ("TimesFM 2.5", carregar_timesfm25, False),
                                        ("TimesFM 3", carregar_timesfm3, False)]:
            try:
                prever = carregador()
                partes.append(rodar(nome, prever, com_covs=covs))
            except Exception as exc:
                print(f"  {nome}: FALHOU ({type(exc).__name__}: {str(exc)[:200]})", file=sys.stderr)
    bt = pd.concat(partes, ignore_index=True)
    bt.to_parquet("backtest_pretreinados.parquet", index=False)
    print("metodos:", sorted(bt["metodo"].unique()), "| linhas:", len(bt))
    if args.so_previsoes:
        sys.exit(0)
    t = avaliar(bt)
    t.to_csv("comparacao_pretreinados.csv", index=False)
    pd.set_option("display.width", 250)
    for c in ["soja", "milho"]:
        x = t[t["commodity"] == c]
        print(f"\n===== {c.upper()} (n={int(x['n'].median())} origens por horizonte)")
        print(x.pivot_table(index="metodo", columns="h", values=["erro", "acerto80", "largura80"]).round(1).to_string())
        nat = x.dropna(subset=["acerto_nativo"])
        if len(nat):
            print("  faixa nativa 10%-90%:")
            print(nat.pivot_table(index="metodo", columns="h", values=["acerto_nativo", "largura_nativa"]).round(0).to_string())
