"""Análise final do TCC: tabelas, testes estatísticos (Friedman/Nemenyi) e figuras.

Uso:  python analise_final.py          (lê ../resultados, grava em ../analise_final)
Protocolo estatístico conforme o projeto de pesquisa (Seção 4.5): Friedman sobre o RMSE (Demšar, 2006),
com pós-teste de Nemenyi. MAE, R², mediana, "sem Brazilian Houses", "só datasets sem falha" e "só rodada 1" são sensibilidade.

Rodada 1: 8 datasets; rodada 2: +4 datasets (kin8nm, news_popularity, santander, black_friday), mesmo protocolo.
Regra para célula (ferramenta × dataset) sem nenhuma execução válida: a ferramenta recebe a PIOR posição do dataset
(empate entre as que falharam), decidida antes de rodar o Friedman. Células com execuções parciais usam a média das válidas.
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parents[1]
RES, OUT = BASE / "resultados", BASE / "analise_final"
FIG, TAB = OUT / "figuras", OUT / "tabelas"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

TOOLS = ["autogluon", "flaml", "h2o", "tpot"]
NOME = {"autogluon": "AutoGluon", "flaml": "FLAML", "h2o": "H2O AutoML", "tpot": "TPOT"}
COR = {"autogluon": "#0072B2", "flaml": "#E69F00", "h2o": "#009E73", "tpot": "#CC79A7"}
# ordem crescente de volume de amostras (igual à Tabela 2 ampliada)
DS = ["concrete", "abalone", "mercedes", "santander", "wine_quality", "kin8nm", "brazilian_houses", "bike_sharing",
      "california", "news_popularity", "diamonds", "black_friday"]
DS_R1 = ["concrete", "abalone", "mercedes", "wine_quality", "brazilian_houses", "bike_sharing", "california", "diamonds"]
NOVOS = ["kin8nm", "news_popularity", "santander", "black_friday"]
DNOME = {"concrete": "Concrete", "abalone": "Abalone", "mercedes": "Mercedes-Benz", "wine_quality": "Wine Quality",
         "brazilian_houses": "Brazilian Houses", "bike_sharing": "Bike Sharing", "california": "California Housing",
         "diamonds": "Diamonds", "kin8nm": "Kin8nm", "news_popularity": "News Popularity", "santander": "Santander",
         "black_friday": "Black Friday"}
BUDGET = 300
N_TENT = 5   # tentativas por célula (seeds 42-46)
GPU_LIMIAR_REAL = 150   # MB; execuções que não são do H2O ficaram todas em 102 MB (sobrecarga de inicialização)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "figure.dpi": 100, "savefig.dpi": 300})


# ---------- formatação pt-BR ----------
def pt(x, d=2):
    # arredonda "meio para cima" (2,125 -> 2,13), como no Excel; o round() do Python arredonda empates ao par (2,12)
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-d)
    s = f"{Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP):,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def dec(x):
    ax = abs(x)
    return 4 if ax < 1 else 3 if ax < 10 else 2 if ax < 100 else 1 if ax < 1000 else 0


def pt_auto(x):
    return pt(x, dec(x))


def pt_min(v, _=None):
    """Número pt-BR sem zeros à direita (para eixos): 6000 -> 6.000 ; 2.15 -> 2,15 ; 6.0 -> 6."""
    s = f"{v:,.4f}".rstrip("0").rstrip(".")
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


# ---------- dados ----------
def carregar():
    linhas = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(RES / "*__seed*.json"))]
    df = pd.DataFrame(linhas)
    df["gpu_usada_mb"] = df["gpu_usada_mb"].fillna(0)
    df["gpu_real"] = df["gpu_usada_mb"] > GPU_LIMIAR_REAL
    df["gpu_flag50"] = df["gpu_usada_mb"] > 50
    # execuções feitas quando as ferramentas ainda dividiam o mesmo processo (rodada 1 inteira + 30 da rodada 2):
    # nelas o "aumento de RAM" cresce com a ordem de execução, então não entra na análise de memória
    f_nao = RES / "rodada2_execucoes_nao_isoladas.txt"
    nao = set(l.strip() for l in open(f_nao, encoding="utf-8") if l.strip() and not l.startswith("#")) if f_nao.exists() else set()
    df["arquivo"] = [f"{r.framework}__{r.dataset}__seed{r.seed}.json" for r in df.itertuples()]
    df["isolada"] = df.dataset.isin(NOVOS) & ~df.arquivo.isin(nao)
    return df


def carregar_falhas():
    """Lê os .falha (execuções que não geraram resultado) e classifica: falhou ou não executada (omitida após 3 falhas)."""
    linhas = []
    for f in sorted(glob.glob(str(RES / "*__seed*.falha"))):
        j = json.load(open(f, encoding="utf-8"))
        erro = str(j.get("erro", ""))
        omitida = j.get("tipo") == "nao_executada" or "omitida para poupar" in erro
        if "limite" in erro and "1200" in erro:
            motivo = "excedeu o limite rígido de 1200 s (encerrada)"
        elif "Not enough time left" in erro:
            motivo = "AutoGluon: tempo insuficiente para o ajuste final (excedeu o orçamento)"
        elif "max_eval_mins" in erro:
            motivo = "TPOT: nenhum indivíduo avaliado na população inicial (limite de 1 min por avaliação)"
        elif "No individuals" in erro:
            motivo = "TPOT: nenhum indivíduo avaliado na população inicial"
        elif "already fit" in erro:
            motivo = "AutoGluon: erro interno (Learner is already fit)"
        elif omitida:
            motivo = "não executada (omitida após falhar nas 3 seeds anteriores)"
        else:
            motivo = erro[:120]
        linhas.append({"ferramenta": j.get("framework", Path(f).name.split("__")[0]), "dataset": j.get("dataset", Path(f).name.split("__")[1]),
                       "seed": int(j.get("seed", Path(f).name.split("seed")[1].split(".")[0])), "situação": "não executada" if omitida else "falhou",
                       "tempo_ate_falhar_s": j.get("tempo_ate_falhar_s"), "motivo": motivo})
    # etapa inicial (8 datasets): as duas falhas do TPOT não geraram arquivo .falha (runner v1)
    for ds, sd in (("diamonds", 45), ("mercedes", 46)):
        linhas.append({"ferramenta": "tpot", "dataset": ds, "seed": sd, "situação": "falhou", "tempo_ate_falhar_s": np.nan,
                       "motivo": "TPOT: nenhum indivíduo avaliado na população inicial (limite de 1 min por avaliação)"})
    return pd.DataFrame(linhas)


def matriz(df, metrica="rmse", agg="mean", drop=()):
    m = df.groupby(["dataset", "framework"])[metrica].agg(agg).unstack()[TOOLS]
    return m.loc[[d for d in DS if d in m.index and d not in drop]]


def ranks(m, maior_melhor=False):
    """Posição por dataset (1 = melhor). Ferramenta sem nenhuma execução válida (NaN) fica na pior posição, com empate entre as que falharam."""
    x = (-m if maior_melhor else m).fillna(np.inf)
    return x.rank(axis=1, method="average")


def datasets_completos(df):
    """Datasets em que as 4 ferramentas têm ao menos uma execução válida."""
    v = df.groupby(["dataset", "framework"]).size().unstack().reindex(columns=TOOLS)
    return [d for d in DS if d in v.index and v.loc[d].notna().all()]


# ---------- testes ----------
def friedman(m, maior_melhor=False):
    N, k = m.shape
    r = ranks(m, maior_melhor)
    chi2, p = stats.friedmanchisquare(*[(-m[t] if maior_melhor else m[t]).fillna(np.inf).values for t in TOOLS])
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pf = stats.f.sf(ff, k - 1, (k - 1) * (N - 1))
    return {"N": N, "k": k, "chi2": chi2, "p": p, "F_ID": ff, "p_ID": pf, "rank_medio": r.mean()}


def nemenyi(rank_medio, N, alpha=0.05):
    k = len(rank_medio)
    se = np.sqrt(k * (k + 1) / (6 * N))
    cd = studentized_range.ppf(1 - alpha, k, 1e6) / np.sqrt(2) * se
    p = pd.DataFrame(index=rank_medio.index, columns=rank_medio.index, dtype=float)
    for a in rank_medio.index:
        for b in rank_medio.index:
            q = abs(rank_medio[a] - rank_medio[b]) / se * np.sqrt(2)
            p.loc[a, b] = 1.0 if a == b else float(studentized_range.sf(q, k, 1e6))
    return cd, p


# ---------- figuras ----------
def salvar(fig, nome):
    fig.savefig(FIG / f"{nome}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{nome}.pdf", bbox_inches="tight")
    plt.close(fig)


def cd_diagram(ax, rank_medio, cd, titulo):
    names = rank_medio.sort_values()
    k = len(names)
    ax.set_xlim(-2.0, k + 2.6)
    ax.set_ylim(-1.95, 1.5)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color="black", lw=1.2)
    for t in range(1, k + 1):
        ax.plot([t, t], [0, 0.08], color="black", lw=1.2)
        ax.text(t, 0.16, str(t), ha="center", va="bottom", fontsize=10)
    ax.text((k + 1) / 2, 1.2, titulo, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.plot([1, 1 + cd], [0.8, 0.8], color="black", lw=2)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.74, 0.86], color="black", lw=2)
    ax.text(1 + cd / 2, 0.9, f"DC = {pt(cd, 2)}", ha="center", va="bottom", fontsize=9)
    half = (k + 1) // 2
    items = list(names.items())
    for i, (nome, r) in enumerate(items[:half]):
        y = -0.7 - 0.5 * i
        ax.plot([r, r, 0.85], [0, y, y], color=COR[nome], lw=1.3)
        ax.text(0.8, y, f"{NOME[nome]} ({pt(r, 2)})", ha="right", va="center", fontsize=10)
    for j, (nome, r) in enumerate(items[half:][::-1]):
        y = -0.7 - 0.5 * j
        ax.plot([r, r, k + 0.15], [0, y, y], color=COR[nome], lw=1.3)
        ax.text(k + 0.2, y, f"{NOME[nome]} ({pt(r, 2)})", ha="left", va="center", fontsize=10)
    vals = names.values
    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i:
            grupos.append((i, j))
    grupos = [g for g in grupos if not any(o != g and o[0] <= g[0] and o[1] >= g[1] for o in grupos)]
    for n, (i, j) in enumerate(grupos):
        y = -0.2 - 0.12 * n
        ax.plot([vals[i] - 0.05, vals[j] + 0.05], [y, y], color="black", lw=4, solid_capstyle="round")


def fig_ranking_barras(r_rmse, N):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ordem = r_rmse.sort_values()
    ax.barh([NOME[t] for t in ordem.index][::-1], ordem.values[::-1], color=[COR[t] for t in ordem.index][::-1])
    for i, v in enumerate(ordem.values[::-1]):
        ax.text(v + 0.04, i, pt(v, 2), va="center", fontweight="bold")
    ax.set_xlim(0, 4.2)
    ax.xaxis.set_major_formatter(FuncFormatter(pt_min))
    ax.set_xlabel(f"Ranking médio do RMSE nos {N} datasets (1 = melhor, 4 = pior)")
    ax.axvline(2.5, color="gray", ls=":", lw=1)
    ax.text(2.52, -0.62, "posição média esperada ao acaso (2,5)", color="gray", fontsize=8, va="top")
    ax.set_ylim(-0.9, 3.5)
    salvar(fig, "fig01_ranking_medio_rmse")


def fig_cd(r_rmse, cd_rmse, r_mae, cd_mae, r_r2, cd_r2):
    legenda = "Ferramentas ligadas por uma barra preta não diferem significativamente (Nemenyi, α = 0,05). DC = distância crítica."
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.4))
    cd_diagram(axs[0], r_rmse, cd_rmse, "RMSE (análise principal)")
    cd_diagram(axs[1], r_mae, cd_mae, "MAE (sensibilidade)")
    cd_diagram(axs[2], r_r2, cd_r2, "R² (sensibilidade)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.5, 0.01, legenda, ha="center", fontsize=8.5, color="#444444")
    salvar(fig, "fig02_diagramas_diferenca_critica")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    cd_diagram(ax, r_rmse, cd_rmse, "RMSE — diagrama de diferença crítica (análise principal)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.5, 0.01, legenda, ha="center", fontsize=8, color="#444444")
    salvar(fig, "fig02a_diagrama_diferenca_critica_rmse")


def fig_heatmap(R, M):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    im = ax.imshow(R.values, cmap="RdYlGn_r", vmin=1, vmax=4, aspect="auto")
    ax.set_xticks(range(len(TOOLS)), [NOME[t] for t in TOOLS])
    ax.set_yticks(range(len(R)), [DNOME[d] for d in R.index])
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            falhou = bool(np.isnan(M.values[i, j]))
            ax.text(j, i, pt(R.values[i, j], 0 if R.values[i, j] % 1 == 0 else 1) + ("*" if falhou else ""), ha="center", va="center",
                    fontweight="bold")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=[1, 2, 3, 4])
    cb.set_label("Posição (1 = menor RMSE)")
    ax.set_title("Posição de cada ferramenta por dataset (RMSE médio das seeds)")
    fig.text(0.5, 0.0, "* sem nenhuma execução válida: recebe a pior posição (empate entre as ferramentas que falharam).",
             ha="center", fontsize=8, color="#444444")
    salvar(fig, "fig03_heatmap_posicao_rmse")


def fig_boxpanels(df, metrica, nome, ylabel, arquivo, titulo):
    fig, axs = plt.subplots(3, 4, figsize=(13, 9.0))
    rng = np.random.default_rng(0)
    for ax, d in zip(axs.ravel(), DS):
        dados = [df[(df.dataset == d) & (df.framework == t)][metrica].values for t in TOOLS]
        pos = [i for i, v in enumerate(dados) if len(v)]
        bp = ax.boxplot([dados[i] for i in pos], positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, i in zip(bp["boxes"], pos):
            patch.set_facecolor(COR[TOOLS[i]]); patch.set_alpha(0.35)
        for i, (t, v) in enumerate(zip(TOOLS, dados)):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COR[t], zorder=3, edgecolor="white", lw=0.4)
            if len(v) == 0:
                ax.text(i, 0.5, "sem\nresultado", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=7.5, color="#B00020")
        ax.set_xlim(-0.6, 3.6)
        ax.set_xticks(range(4), [("AG" if t == "autogluon" else "FLAML" if t == "flaml" else "H2O" if t == "h2o" else "TPOT") +
                                 (f"\n(n={len(v)})" if 0 < len(v) < N_TENT else "") for t, v in zip(TOOLS, dados)], fontsize=8)
        ax.set_title(DNOME[d], fontsize=10)
        if d == "brazilian_houses":
            ax.set_yscale("log")
            ax.set_title("Brazilian Houses (escala log)", fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(pt_min))
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(titulo, fontsize=12, y=1.0)
    fig.text(0.005, 0.5, ylabel, rotation=90, va="center")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COR[t], label=NOME[t]) for t in TOOLS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.015, 0.02, 1, 0.98))
    salvar(fig, arquivo)


def fig_relativo(m):
    rel = m.div(m.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ys = np.arange(len(rel))[::-1]
    for j, t in enumerate(TOOLS):
        ax.scatter(rel[t].values, ys + (j - 1.5) * 0.12, s=60, color=COR[t], label=NOME[t], zorder=3, edgecolor="white", lw=0.5)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.97, 12)
    ax.set_xticks([1, 1.1, 1.5, 2, 3, 5, 10], ["1", "1,1", "1,5", "2", "3", "5", "10"])
    ax.text(0.99, 0.005, "ferramentas sem execução válida não aparecem (Santander: AutoGluon e TPOT; News Popularity: AutoGluon)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    ax.minorticks_off()
    ax.set_yticks(ys, [DNOME[d] for d in rel.index])
    ax.set_xlabel("RMSE médio ÷ RMSE da melhor ferramenta no dataset (1 = melhor; escala log)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.12))
    ax.grid(axis="x", alpha=0.3)
    salvar(fig, "fig05_rmse_relativo_ao_melhor")


def fig_custo(df):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4))
    cfg = [("tempo_real_s", "Tempo real de execução (s)\n(todas as execuções válidas)"),
           ("ram_pico_mb", "Pico de RAM (MB)\n(só execuções em processo isolado)"),
           ("ram_delta_mb", "Aumento de RAM durante a execução (MB)\n(só execuções em processo isolado)")]
    for ax, (col, tit) in zip(axs, cfg):
        base = df if col == "tempo_real_s" else df[df.isolada]
        dados = [base[base.framework == t][col].values for t in TOOLS]
        bp = ax.boxplot(dados, widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=3, alpha=0.5), medianprops=dict(color="black", lw=1.5))
        for patch, t in zip(bp["boxes"], TOOLS):
            patch.set_facecolor(COR[t]); patch.set_alpha(0.55)
        ax.set_xticks(range(1, 5), [f"{NOME[t].replace(' AutoML', '')}\nmed. {pt(np.median(v), 0)}\n(n={len(v)})" for t, v in zip(TOOLS, dados)], fontsize=8)
        ax.set_title(tit, fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(pt_min))
        if col == "tempo_real_s":
            ymax = 520
            ax.set_ylim(285, ymax)
            ax.axhline(BUDGET, color="red", ls="--", lw=1)
            ax.text(4.5, BUDGET + 3, "orçamento (300 s)", color="red", fontsize=8, va="bottom", ha="right")
            for i, v in enumerate(dados, 1):
                fora = v[v > ymax]
                if len(fora):
                    ax.annotate(NOME[TOOLS[i - 1]] + ": " + ", ".join(pt(x, 0) for x in sorted(fora)) + " s\n(fora da escala)", (i, ymax),
                                xytext=(i + 0.3, ymax - 14), fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"), va="top")
    salvar(fig, "fig06_custo_computacional")


def fig_tradeoff(rank_rmse, df):
    iso = df[df.isolada]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for t in TOOLS:
        v = iso[iso.framework == t]["ram_delta_mb"]
        x = v.median()
        ax.errorbar(x, rank_rmse[t], xerr=[[x - v.min()], [v.max() - x]], fmt="o", ms=11, color=COR[t], capsize=3, zorder=3)
        ax.annotate(f"{NOME[t]} (n={len(v)})", (x, rank_rmse[t]), textcoords="offset points", xytext=(10, -16 if t == "autogluon" else 9), fontsize=10)
    ax.set_ylim(3.55, 1.85)
    ax.set_xlim(300, 4100)
    ax.set_xlabel("Aumento de RAM (MB): mediana e amplitude (mín.–máx.)")
    ax.set_ylabel("Ranking médio do RMSE (1 = melhor)")
    ax.set_title("Desempenho preditivo × custo de memória\n(canto superior esquerdo = melhor)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 0)))
    ax.yaxis.set_major_formatter(FuncFormatter(pt_min))
    ax.text(0.99, 0.02, "RAM: apenas execuções da rodada 2 feitas em processo isolado", transform=ax.transAxes,
            ha="right", fontsize=7.5, color="#555555")
    salvar(fig, "fig07_tradeoff_desempenho_ram")


def fig_gpu(df):
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(4)
    p50 = [100 * df[df.framework == t].gpu_flag50.mean() for t in TOOLS]
    pr = [100 * df[df.framework == t].gpu_real.mean() for t in TOOLS]
    ax.bar(x, p50, width=0.6, color="#cccccc", label="marcadas com limiar de 50 MB (inclui sobrecarga de ~102 MB)")
    ax.bar(x, pr, width=0.6, color="#2F7D3C", label=f"uso real de GPU (> {GPU_LIMIAR_REAL} MB)")
    for i, t in enumerate(TOOLS):
        n = (df.framework == t).sum()
        ax.text(i, max(p50[i], pr[i]) + 2, f"{int(df[df.framework == t].gpu_real.sum())}/{n}", ha="center", fontweight="bold")
    ax.set_xticks(x, [NOME[t] for t in TOOLS])
    ax.set_ylabel("% das execuções com uso de GPU")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.2))
    salvar(fig, "fig08_uso_gpu")


def fig_brazilian(df):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    for t in TOOLS:
        s = df[(df.dataset == "brazilian_houses") & (df.framework == t)].sort_values("seed")
        ax.plot(s.seed, s.rmse, marker="o", color=COR[t], label=NOME[t], lw=1.5)
    ax.set_yscale("log")
    ax.set_xticks(range(42, 47))
    ax.set_xlabel("Seed (define a partição treino/teste)")
    ax.set_ylabel("RMSE (escala log)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 0)))
    ax.set_title("(a) RMSE por seed e ferramenta")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axs[1]
    maxy = []
    for s in range(42, 47):
        v = np.nan
        for t in TOOLS:
            f = RES / f"{t}__brazilian_houses__seed{s}_pred.npz"
            if f.exists():
                v = float(np.max(np.load(f)["y"])); break
        maxy.append(v)
    ax.bar(range(42, 47), maxy, color=["#d95f02" if v > 100000 else "#999999" for v in maxy])
    ax.set_yscale("log")
    ax.set_xticks(range(42, 47))
    ax.set_xlabel("Seed")
    ax.set_ylabel("Maior aluguel no conjunto de teste (R$, log)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 0)))
    for s, v in zip(range(42, 47), maxy):
        ax.text(s, v * 1.15, pt(v, 0), ha="center", fontsize=8)
    ax.set_ylim(top=max(maxy) * 5)
    ax.set_title("(b) Valor extremo presente no teste")
    fig.tight_layout()
    salvar(fig, "fig09_brazilian_houses")
    return dict(zip(range(42, 47), maxy))


# ---------- tabelas ----------
def tabela_desempenho(df, metrica):
    g = df.groupby(["dataset", "framework"])[metrica]
    mean, std, n = g.mean(), g.std(), g.count()
    out = pd.DataFrame(index=[DNOME[d] for d in DS], columns=[NOME[t] for t in TOOLS], dtype=object)
    num = pd.DataFrame(index=[DNOME[d] for d in DS], columns=[f"{NOME[t]} {s}" for t in TOOLS for s in ("média", "dp", "n")], dtype=float)
    for d in DS:
        md = pd.Series({t: mean.get((d, t), np.nan) for t in TOOLS})
        melhor = (md.idxmax() if metrica == "r2" else md.idxmin())
        for t in TOOLS:
            mu, sd, nn = mean.get((d, t), np.nan), std.get((d, t), np.nan), int(n.get((d, t), 0))
            if nn == 0:
                txt = "sem resultado (0/5)"
            else:
                txt = f"{pt(mu, dec(mu))} ± {pt(sd, dec(mu))}" if nn > 1 else f"{pt(mu, dec(mu))}"
                txt += (f" (n={nn})" if nn != N_TENT else "")
            out.loc[DNOME[d], NOME[t]] = ("★ " if (t == melhor and nn > 0) else "") + txt
            num.loc[DNOME[d], [f"{NOME[t]} média", f"{NOME[t]} dp", f"{NOME[t]} n"]] = [mu, sd, nn]
    return out, num


def salvar_excel(abas):
    with pd.ExcelWriter(OUT / "tabelas_monografia.xlsx", engine="openpyxl") as w:
        for nome, d in abas.items():
            d.to_excel(w, sheet_name=nome[:31])
            ws = w.sheets[nome[:31]]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(48, max(12, max(len(str(c.value or "")) for c in col) + 2))
            from openpyxl.styles import Font, PatternFill, Alignment
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2F7D3C")
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            ws.freeze_panes = "B2"


def main():
    df = carregar()
    falhas = carregar_falhas()
    n_exec = len(df)
    n_falhou = int((falhas["situação"] == "falhou").sum()); n_omit = int((falhas["situação"] == "não executada").sum())
    print(f"Execuções válidas: {n_exec}/{len(DS) * len(TOOLS) * N_TENT} | falharam: {n_falhou} | não executadas (omitidas): {n_omit}")
    completos = datasets_completos(df)
    com_falha = [d for d in DS if d not in completos]
    print("Datasets com as 4 ferramentas válidas:", len(completos), "| com ferramenta sem resultado:", com_falha)

    # --- análise principal (projeto 4.5): Friedman sobre o RMSE médio por dataset; célula sem resultado = pior posição ---
    M = matriz(df, "rmse", "mean")
    R = ranks(M)
    fr = friedman(M)
    cd, pn = nemenyi(fr["rank_medio"], fr["N"])
    # --- sensibilidade ---
    drop_falhas = tuple(com_falha)
    var = {
        "RMSE médio (análise principal)": (matriz(df, "rmse", "mean"), False),
        "RMSE mediano": (matriz(df, "rmse", "median"), False),
        "MAE médio": (matriz(df, "mae", "mean"), False),
        "R² médio": (matriz(df, "r2", "mean"), True),
        "RMSE médio sem Brazilian Houses": (matriz(df, "rmse", "mean", drop=("brazilian_houses",)), False),
        "RMSE médio só datasets sem falha": (matriz(df, "rmse", "mean", drop=drop_falhas), False),
        "RMSE médio só rodada 1 (8 datasets)": (matriz(df, "rmse", "mean", drop=tuple(NOVOS)), False),
    }
    sens, resf = [], {}
    for nome, (m, mb) in var.items():
        f = friedman(m, mb)
        resf[nome] = f
        sens.append({"Análise": nome, "N datasets": f["N"], "χ² Friedman": f["chi2"], "p (Friedman)": f["p"],
                     "F Iman-Davenport": f["F_ID"], "p (Iman-Davenport)": f["p_ID"],
                     "DC Nemenyi": nemenyi(f["rank_medio"], f["N"])[0],
                     **{f"Ranking {NOME[t]}": f["rank_medio"][t] for t in TOOLS}})
    sens = pd.DataFrame(sens).set_index("Análise")
    cd_mae, pn_mae = nemenyi(resf["MAE médio"]["rank_medio"], resf["MAE médio"]["N"])
    cd_r2, pn_r2 = nemenyi(resf["R² médio"]["rank_medio"], resf["R² médio"]["N"])
    cd_med, pn_med = nemenyi(resf["RMSE mediano"]["rank_medio"], resf["RMSE mediano"]["N"])
    cd_sf, pn_sf = nemenyi(resf["RMSE médio só datasets sem falha"]["rank_medio"], resf["RMSE médio só datasets sem falha"]["N"])

    # --- tabelas ---
    abas = {}
    for met, nome in (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R2")):
        txt, num = tabela_desempenho(df, met)
        abas[nome] = txt
        txt.to_csv(TAB / f"desempenho_{met}.csv", encoding="utf-8-sig")
        num.to_csv(TAB / f"desempenho_{met}_numerico.csv", encoding="utf-8-sig")
    rk = R.copy(); rk.index = [DNOME[d] for d in rk.index]; rk.columns = [NOME[t] for t in TOOLS]
    rk.loc["Ranking médio"] = rk.mean()
    abas["Ranking RMSE"] = rk.round(3)
    rk.to_csv(TAB / "ranking_rmse.csv", encoding="utf-8-sig")
    rel = M.div(M.min(axis=1), axis=0)
    geo = np.exp(np.log(rel).mean())
    geo_sem = np.exp(np.log(rel.drop("brazilian_houses")).mean())

    tp = pd.DataFrame({
        "χ² Friedman": [fr["chi2"]], "gl": [fr["k"] - 1], "p (Friedman)": [fr["p"]],
        "F Iman-Davenport": [fr["F_ID"]], "p (Iman-Davenport)": [fr["p_ID"]], "N datasets": [fr["N"]],
        "k ferramentas": [fr["k"]], "Distância crítica Nemenyi (α=0,05)": [cd],
        "Decisão (α=0,05)": ["H0 não rejeitada" if fr["p"] >= 0.05 else "H0 rejeitada"]}).T
    tp.columns = ["RMSE (análise principal)"]
    abas["Testes RMSE"] = tp
    pnn = pn.copy(); pnn.index = pnn.columns = [NOME[t] for t in TOOLS]
    abas["Nemenyi RMSE (p)"] = pnn.round(4)
    pnm = pn_mae.copy(); pnm.index = pnm.columns = [NOME[t] for t in TOOLS]
    abas["Nemenyi MAE (p)"] = pnm.round(4)
    pnr = pn_r2.copy(); pnr.index = pnr.columns = [NOME[t] for t in TOOLS]
    abas["Nemenyi R2 (p)"] = pnr.round(4)
    pnd = pn_med.copy(); pnd.index = pnd.columns = [NOME[t] for t in TOOLS]
    abas["Nemenyi RMSE mediano (p)"] = pnd.round(4)
    pns = pn_sf.copy(); pns.index = pns.columns = [NOME[t] for t in TOOLS]
    abas["Nemenyi só sem falha (p)"] = pns.round(4)
    abas["Sensibilidade"] = sens.round(4)
    sens.to_csv(TAB / "sensibilidade.csv", encoding="utf-8-sig")

    custo = []
    for t in TOOLS:
        s = df[df.framework == t]
        si = s[s.isolada]
        custo.append({"Ferramenta": NOME[t], "Execuções": len(s), "Tempo médio (s)": s.tempo_real_s.mean(),
                      "Tempo mediano (s)": s.tempo_real_s.median(), "Tempo máximo (s)": s.tempo_real_s.max(),
                      "% execuções > 330 s (10% acima do orçamento)": 100 * (s.tempo_real_s > 1.1 * BUDGET).mean(),
                      "Execuções em processo isolado (RAM confiável)": len(si),
                      "RAM pico mediana, isoladas (MB)": si.ram_pico_mb.median(),
                      "RAM pico mín.–máx., isoladas (MB)": f"{si.ram_pico_mb.min():.0f}–{si.ram_pico_mb.max():.0f}",
                      "Aumento de RAM mediano, isoladas (MB)": si.ram_delta_mb.median(),
                      "Aumento de RAM mín.–máx., isoladas (MB)": f"{si.ram_delta_mb.min():.0f}–{si.ram_delta_mb.max():.0f}",
                      "(ref. contaminada) RAM pico mediana, todas (MB)": s.ram_pico_mb.median(),
                      "(ref. contaminada) Aumento de RAM mediano, todas (MB)": s.ram_delta_mb.median()})
    custo = pd.DataFrame(custo).set_index("Ferramenta")
    abas["Custo computacional"] = custo.round(1)
    custo.round(2).to_csv(TAB / "custo_computacional.csv", encoding="utf-8-sig")

    iso = df[df.isolada]
    comp = iso.pivot_table(index=["dataset", "seed"], columns="framework", values="ram_delta_mb").reindex(columns=TOOLS).dropna()
    comp.index = [f"{DNOME[d]} (seed {sd})" for d, sd in comp.index]
    comp.columns = [NOME[t] for t in TOOLS]
    comp.round(0).to_csv(TAB / "ram_grupos_completos_isolados.csv", encoding="utf-8-sig")
    abas["RAM grupos isolados"] = comp.round(0)
    cel = df.groupby(["dataset", "framework"]).size().unstack().reindex(index=DS, columns=TOOLS).fillna(0).astype(int)
    cel.index = [DNOME[d] for d in cel.index]; cel.columns = [NOME[t] for t in TOOLS]
    cel.loc["Total"] = cel.sum()
    cel.to_csv(TAB / "execucoes_validas_por_celula.csv", encoding="utf-8-sig")
    abas["Execuções válidas (de 5)"] = cel

    gpu = []
    for t in TOOLS:
        s = df[df.framework == t]
        gpu.append({"Ferramenta": NOME[t], "Execuções": len(s), "GPU disponível (execuções)": int(s.gpu_disponivel.sum()),
                    f"Uso real (> {GPU_LIMIAR_REAL} MB)": int(s.gpu_real.sum()), "Marcadas com limiar original (> 50 MB)": int(s.gpu_flag50.sum()),
                    "Memória GPU média nas execuções (MB)": s.gpu_usada_mb.mean(), "Maior valor (MB)": s.gpu_usada_mb.max()})
    gpu = pd.DataFrame(gpu).set_index("Ferramenta")
    abas["Uso de GPU"] = gpu.round(1)
    gpu.round(1).to_csv(TAB / "uso_gpu.csv", encoding="utf-8-sig")

    cv = (df.groupby(["dataset", "framework"]).rmse.std() / df.groupby(["dataset", "framework"]).rmse.mean()).unstack()[TOOLS]
    sintese = pd.DataFrame({
        "Ranking médio RMSE": R.mean(),
        "Ranking médio MAE": ranks(matriz(df, "mae")).mean(),
        "Ranking médio R²": ranks(matriz(df, "r2"), True).mean(),
        f"Vezes em 1º (RMSE, de {len(DS)})": (R == 1).sum(),
        "RMSE relativo ao melhor (média geométrica)": geo,
        "idem, sem Brazilian Houses": geo_sem,
        "CV mediano do RMSE entre seeds (%)": 100 * cv.median(),
        "Tempo médio (s)": df.groupby("framework").tempo_real_s.mean(),
        "Aumento de RAM mediano, só execuções isoladas (MB)": df[df.isolada].groupby("framework").ram_delta_mb.median(),
        "Execuções com uso real de GPU": df.groupby("framework").gpu_real.sum(),
        "Execuções válidas": df.groupby("framework").size(),
    }).T
    sintese.columns = [NOME[t] for t in TOOLS]
    abas["Síntese"] = sintese.round(3)
    sintese.round(3).to_csv(TAB / "sintese_final.csv", encoding="utf-8-sig")
    dsmeta = pd.DataFrame({
        "Dataset": [DNOME[d] for d in DS],
        "Rodada": [2 if d in NOVOS else 1 for d in DS],
        "Linhas": [1030, 4177, 4209, 4459, 6497, 8192, 10692, 17379, 20640, 39644, 53940, 166821],
        "Atributos": [8, 8, 376, 4991, 11, 8, 12, 12, 8, 59, 9, 9],
        "% categóricos": [0, 12.5, 2.1, 0, 0, 0, 33.3, 33.3, 0, 0, 33.3, 44.4],
        "% zeros nos atributos numéricos": [17.18, 0.01, 84.23, 96.9, 0.21, 0.0, 7.78, 10.13, 0.0, 28.2, 0.01, 14.5],
        "Cardinalidade máx. categórica": [0, 3, 47, 0, 0, 0, 35, 4, 0, 0, 8, 7]}).set_index("Dataset")
    abas["Datasets"] = dsmeta
    dsmeta.to_csv(TAB / "datasets.csv", encoding="utf-8-sig")
    abas["Falhas"] = falhas.set_index("ferramenta")
    falhas.to_csv(TAB / "falhas.csv", index=False, encoding="utf-8-sig")
    salvar_excel(abas)

    # --- figuras ---
    rm, rmae, rr2 = resf["RMSE médio (análise principal)"]["rank_medio"], resf["MAE médio"]["rank_medio"], resf["R² médio"]["rank_medio"]
    fig_ranking_barras(rm, fr["N"])
    fig_cd(rm, cd, rmae, cd_mae, rr2, cd_r2)
    fig_heatmap(R, M)
    fig_boxpanels(df, "rmse", "RMSE", "RMSE (menor é melhor)", "fig04_rmse_por_dataset", "RMSE por dataset: cada ponto é uma seed (5 repetições)")
    fig_relativo(M)
    fig_boxpanels(df, "mae", "MAE", "MAE (menor é melhor)", "fig04b_mae_por_dataset", "MAE por dataset: cada ponto é uma seed (5 repetições)")
    fig_custo(df)
    fig_tradeoff(rm, df)
    fig_gpu(df)
    maxy = fig_brazilian(df)

    # --- resumo no console ---
    print("\n=== ANÁLISE PRINCIPAL (RMSE médio por dataset) ===")
    print("Ranking médio:", {NOME[t]: round(fr['rank_medio'][t], 3) for t in TOOLS})
    print(f"Friedman: χ²={fr['chi2']:.3f} (gl={fr['k']-1}) p={fr['p']:.4f} | Iman-Davenport F={fr['F_ID']:.3f} p={fr['p_ID']:.4f}")
    print(f"Nemenyi DC(α=0,05)={cd:.3f}")
    print("\n=== SENSIBILIDADE ===")
    print(sens.round(4).to_string())
    print(f"\nNemenyi MAE: DC={cd_mae:.3f} | R²: DC={cd_r2:.3f} | só sem falha: DC={cd_sf:.3f}")
    for nome, p_, r_ in (("RMSE", pn, rm), ("RMSE mediano", pn_med, rm), ("MAE", pn_mae, rmae), ("R²", pn_r2, rr2), ("RMSE só sem falha", pn_sf, rm)):
        pares = [(NOME[a], NOME[b], p_.loc[a, b]) for i, a in enumerate(TOOLS) for b in TOOLS[i + 1:] if p_.loc[a, b] < 0.05]
        print(f"Pares significativos Nemenyi ({nome}):", pares or "nenhum")
    print("\n=== SÍNTESE ===")
    print(sintese.round(3).to_string())
    print("\nFalhas:\n", falhas.groupby(["ferramenta", "dataset", "situação"]).size().to_string())
    print("\nRAM (isoladas, grupos completos):\n", comp.to_string())
    print("\nMaior alvo no teste por seed (Brazilian):", {k: round(v) for k, v in maxy.items()})
    print("\nGPU:\n", gpu.to_string())
    print("\nCusto:\n", custo.round(1).to_string())
    print(f"\nArquivos em: {OUT}")


if __name__ == "__main__":
    main()
