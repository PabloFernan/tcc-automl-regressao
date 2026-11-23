"""Análise suplementar do Brazilian Houses: avalia as previsões já salvas SEM as linhas de teste que são cópias exatas
de linhas do treino (vazamento por duplicatas). Não retreina nada: reconstrói a partição de cada seed e filtra o teste.

Uso: python analise_brazilian.py   (precisa de internet na 1ª vez para baixar o dataset do OpenML)
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets_meta import carregar
from analise_final import NOME, COR, TOOLS, RES, FIG, TAB, pt, pt_min, salvar

X, y = carregar("brazilian_houses")
df = X.copy()
df["__y__"] = y.values
h = pd.util.hash_pandas_object(df, index=False)
print(f"Linhas: {len(df)} | duplicatas exatas (X+y): {int(h.duplicated().sum())} ({100*h.duplicated().mean():.1f}%)")

linhas, info = [], []
for seed in range(42, 47):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed)
    vazou = h.loc[Xte.index].isin(set(h.loc[Xtr.index])).values
    info.append({"seed": seed, "linhas_teste": len(yte), "teste_duplicadas_do_treino": int(vazou.sum()),
                 "maior_alvo_teste": float(yte.max()), "maior_alvo_vazou": bool(vazou[np.argmax(yte.values)])})
    for t in TOOLS:
        f = RES / f"{t}__brazilian_houses__seed{seed}_pred.npz"
        if not f.exists():
            continue
        z = np.load(f)
        assert np.allclose(z["y"], yte.values), f"partição não bate: {t} seed {seed}"
        e = z["p"] - z["y"]
        linhas.append({"seed": seed, "ferramenta": t,
                       "rmse_todas": float(np.sqrt(np.mean(e ** 2))), "mae_todas": float(np.mean(np.abs(e))),
                       "rmse_sem_dup": float(np.sqrt(np.mean(e[~vazou] ** 2))), "mae_sem_dup": float(np.mean(np.abs(e[~vazou])))})
res = pd.DataFrame(linhas)
info = pd.DataFrame(info)
print("\nVazamento por seed (linhas do teste idênticas a linhas do treino):")
print(info.to_string(index=False))
print("\nCobertura de previsões salvas (execuções com npz):", res.groupby("ferramenta").size().to_dict())

agg = res.groupby("ferramenta")[["rmse_todas", "rmse_sem_dup", "mae_todas", "mae_sem_dup"]].mean().loc[[t for t in TOOLS if t in res.ferramenta.unique()]]
agg["n_seeds"] = res.groupby("ferramenta").size()
agg["rank_todas"] = agg.rmse_todas.rank()
agg["rank_sem_dup"] = agg.rmse_sem_dup.rank()
agg.index = [NOME[t] for t in agg.index]
print("\nMédia entre seeds (somente execuções com previsões salvas):")
print(agg.round(3).to_string())

res.to_csv(TAB / "brazilian_sem_vazamento_por_execucao.csv", index=False, encoding="utf-8-sig")
agg.round(3).to_csv(TAB / "brazilian_sem_vazamento_resumo.csv", encoding="utf-8-sig")
info.to_csv(TAB / "brazilian_vazamento_por_seed.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(agg))
w = 0.38
ax.bar(x - w / 2, agg.rmse_todas, w, color="#bbbbbb", label="teste completo")
ax.bar(x + w / 2, agg.rmse_sem_dup, w, color=[COR[t] for t in TOOLS if NOME[t] in agg.index], label="sem linhas duplicadas do treino")
ax.set_yscale("log")
ax.set_xticks(x, agg.index)
ax.yaxis.set_major_formatter(FuncFormatter(pt_min))
ax.set_ylabel("RMSE médio entre seeds (escala log)")
ax.set_title("Brazilian Houses: efeito das duplicatas treino/teste no RMSE", fontsize=11)
for i, (a, b) in enumerate(zip(agg.rmse_todas, agg.rmse_sem_dup)):
    ax.text(i - w / 2, a * 1.1, pt(a, 0), ha="center", fontsize=8)
    ax.text(i + w / 2, b * 1.1, pt(b, 0), ha="center", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper right")
salvar(fig, "fig10_brazilian_sem_vazamento")
print(f"\nFigura: {FIG / 'fig10_brazilian_sem_vazamento.png'}")
