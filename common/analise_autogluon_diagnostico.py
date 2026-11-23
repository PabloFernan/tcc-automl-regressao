"""Figura/tabela do diagnóstico do AutoGluon no Brazilian Houses (variantes rodadas no Colab, seeds 44 e 45).
Entrada: analise_final/tabelas/diagnostico_autogluon_variantes.csv (resumo_variantes.csv do notebook de diagnóstico)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analise_final import RES, TAB, COR, pt, pt_min, salvar

df = pd.read_csv(TAB / "diagnostico_autogluon_variantes.csv")
ORD = ["base", "sem_nn_knn", "log1p"]
NOMEV = {"base": "Configuração do estudo", "sem_nn_knn": "Sem rede neural e KNN", "log1p": "Alvo em log1p"}
COREV = {"base": "#9a9a9a", "sem_nn_knn": "#2F7D3C", "log1p": "#6a51a3"}
SEEDS = sorted(df.seed.unique())


def ref(tool, seed, col):
    f = RES / f"{tool}__brazilian_houses__seed{seed}.json"
    return json.load(open(f, encoding="utf-8"))[col]


fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
paineis = [("rmse", "RMSE", False), ("mae", "MAE", False), ("maior_superestimativa", "Maior superestimativa (R$, escala log)", True)]
for ax, (col, titulo, log) in zip(axs, paineis):
    for gi, s in enumerate(SEEDS):
        for vi, v in enumerate(ORD):
            val = float(df[(df.seed == s) & (df.variante == v)][col].iloc[0])
            x = gi + (vi - 1) * 0.27
            ax.bar(x, val, 0.25, color=COREV[v], label=NOMEV[v] if gi == 0 else None)
            ax.text(x, val * (1.12 if log else 1.0) + (0 if log else ax.get_ylim()[1] * 0.0), pt(val, 0), ha="center", va="bottom", fontsize=7.5, rotation=90)
        if col in ("rmse", "mae"):
            for tool in ("flaml", "h2o"):
                y = ref(tool, s, col)
                ax.plot([gi - 0.42, gi + 0.42], [y, y], color=COR[tool], lw=2, ls="--", label=("referência: " + ("FLAML" if tool == "flaml" else "H2O") + " no estudo") if gi == 0 else None)
    ax.set_xticks(range(len(SEEDS)), [f"seed {s}" for s in SEEDS])
    ax.set_title(titulo, fontsize=10)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(500, ax.get_ylim()[1] * 12)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(pt_min))
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("AutoGluon no Brazilian Houses: efeito de cada intervenção (5 min por execução)", fontsize=11.5, y=1.01)
fig.tight_layout(rect=(0, 0.04, 1, 1))
salvar(fig, "fig11_autogluon_diagnostico")

# tabela de reducoes
t = df.pivot(index="seed", columns="variante", values=["rmse", "mae", "maior_superestimativa"])
linhas = []
for s in SEEDS:
    for v in ("sem_nn_knn", "log1p"):
        linhas.append({"seed": s, "variante": NOMEV[v],
                       "RMSE: variação vs base (%)": 100 * (t[("rmse", v)][s] / t[("rmse", "base")][s] - 1),
                       "MAE: variação vs base (%)": 100 * (t[("mae", v)][s] / t[("mae", "base")][s] - 1),
                       "Maior superestimativa: base → variante (R$)": f"{pt(t[('maior_superestimativa', 'base')][s], 0)} → {pt(t[('maior_superestimativa', v)][s], 0)}"})
red = pd.DataFrame(linhas)
red.round(1).to_csv(TAB / "diagnostico_autogluon_reducoes.csv", index=False, encoding="utf-8-sig")
print(red.round(1).to_string(index=False))
