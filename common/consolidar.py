"""Consolida os JSON de resultados/: média ± desvio, mediana, ranking e Friedman/Nemenyi.

Uso:  python consolidar.py [pasta_com_json]    (padrão: resultados)
Ignora subpastas (ex.: sem_catboost) e arquivos que não sejam {tool}__{dataset}__seed{n}.json.
Saídas na pasta: tabela_resumo.csv, tabela_resumo.md, ranking.csv, estatistica.txt
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

METRICAS = ["rmse", "mae", "r2", "tempo_real_s", "ram_pico_mb", "ram_delta_mb"]


def carregar(pasta):
    linhas = []
    for f in sorted(glob.glob(os.path.join(pasta, "*__*__seed*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        r["gpu_disp"] = r.get("gpu_disponivel")
        linhas.append(r)
    return pd.DataFrame(linhas)


def resumo(df):
    g = df.groupby(["dataset", "framework"])
    cols = [m for m in METRICAS if m in df.columns]
    out = g[cols].agg(["mean", "std", "median"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g.size()
    out["gpu_usada_em"] = g["gpu_usada"].sum() if "gpu_usada" in df.columns else np.nan
    return out.reset_index()


def ranking(df):
    """Ranking médio por dataset (1 = menor RMSE médio) e ranking médio geral."""
    m = df.groupby(["dataset", "framework"])["rmse"].mean().unstack()
    rk = m.rank(axis=1, method="average")
    return m, rk


def friedman(m):
    """Friedman sobre a matriz dataset x ferramenta (RMSE médio). Exige >=3 ferramentas e >=2 datasets."""
    from scipy import stats
    m = m.dropna()
    txt = [f"Datasets completos (todas as ferramentas): {len(m)} -> {list(m.index)}",
           f"Ferramentas: {list(m.columns)}"]
    if m.shape[1] < 3 or m.shape[0] < 2:
        return "\n".join(txt + ["Friedman não aplicável (precisa >=3 ferramentas e >=2 datasets)."]), None
    if m.shape[0] < 6:
        txt.append("AVISO: menos de 6 datasets; resultado descritivo, sem poder estatístico.")
    st, p = stats.friedmanchisquare(*[m[c].values for c in m.columns])
    txt.append(f"Friedman: estatística={st:.4f}  p={p:.4g}")
    pos = None
    if p < 0.05:
        try:
            import scikit_posthocs as sp
            pos = sp.posthoc_nemenyi_friedman(m.values)
            pos.index = pos.columns = list(m.columns)
            txt.append("Nemenyi (p-valores):\n" + pos.round(4).to_string())
        except ImportError:
            txt.append("Para o pós-teste de Nemenyi: pip install scikit-posthocs")
    else:
        txt.append("Hipótese nula não rejeitada (p>=0,05): sem diferença global significativa.")
    return "\n".join(txt), pos


def main(pasta):
    df = carregar(pasta)
    if df.empty:
        sys.exit(f"Nenhum JSON encontrado em {pasta}")
    res = resumo(df)
    res.to_csv(os.path.join(pasta, "tabela_resumo.csv"), index=False)
    m, rk = ranking(df)
    rk_med = rk.mean().sort_values()
    rk.assign(**{"_": np.nan}).drop(columns="_").to_csv(os.path.join(pasta, "ranking.csv"))
    txt, _ = friedman(m)
    txt = (f"Execuções lidas: {len(df)} | seeds por combinação: "
           f"{df.groupby(['dataset','framework']).size().min()}–{df.groupby(['dataset','framework']).size().max()}\n\n"
           f"Ranking médio (RMSE, menor é melhor):\n{rk_med.round(3).to_string()}\n\n{txt}\n")
    open(os.path.join(pasta, "estatistica.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(res[["dataset", "framework", "rmse_mean", "rmse_std", "tempo_real_s_mean", "n_seeds"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados")
