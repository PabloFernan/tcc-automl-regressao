"""Catálogo de datasets candidatos (Tabela 2) e extração de metadados para justificar a escolha.

Critérios da banca: variar volume de amostras, nº de atributos, atributos categóricos e esparsidade.
Os IDs OpenML abaixo foram conferidos (nome, shape e alvo) em 2026-11; rode `python datasets_meta.py` para regerar os metadados.
"""
import json, os, sys
import pandas as pd

# nome -> (fonte, identificador, coluna alvo ou None = alvo padrão do OpenML)
CANDIDATOS = {
    # --- rodada 1 (8 datasets): cobrem volume, dimensionalidade, categóricos e esparsidade
    "abalone":       ("openml", 183, None),
    "concrete":      ("openml", 4353, None),   # alvo padrão veio NaN: o loader usa a última coluna
    "wine_quality":  ("openml", 287, None),
    "california":    ("sklearn", "california_housing", None),
    "bike_sharing":  ("openml", 42712, None),
    "brazilian_houses": ("openml", 42688, None),
    "diamonds":      ("openml", 42225, None),
    "mercedes":      ("openml", 42570, None),
    # --- rodada 2 (+4 datasets, de trabalhos relacionados): preenchem as lacunas da rodada 1
    "kin8nm":          ("openml", 189, None),     # Hanussek et al. (2020); 8.192 x 8, numérico denso, não linear. ATENÇÃO: o ID 807 é a versão de CLASSIFICAÇÃO (alvo binaryClass)
    "news_popularity": ("openml", 42724, None),   # Dissanayake et al. (2025); 39.644 x 59, numérico com muitos atributos binários/zeros
    "santander":       ("openml", 42572, None),   # Gijsbers et al. (2024); 4.459 x 4.991 (p >> n), 96,9% de zeros
    "black_friday":    ("openml", 41540, None),   # Gijsbers et al. (2024); 166.821 x 9, 44% categóricos, maior volume
    # descartados: houses(537) = duplicata de california; boston = ressalva ética; year(44027) = 515 mil linhas (inviável em 300 s/Colab e TPOT);
    # Rossmann, Flight Delay, Video Game Sales, bases de baterias/geotécnicas/materiais = não disponíveis em repositório aberto de uso direto
}


def carregar(nome):
    fonte, ident, alvo = CANDIDATOS[nome]
    if fonte == "sklearn":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        return d.data, d.target
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=ident, as_frame=True, parser="auto")
    if alvo is not None:
        X, y = d.frame.drop(columns=[alvo]), d.frame[alvo]
    else:
        X, y = d.data, d.target
    if y is not None:
        y = pd.to_numeric(y, errors="coerce")
    if y is None or y.isna().all():  # sem alvo padrão ou não numérico -> usa a última coluna
        X, y = d.frame.iloc[:, :-1], pd.to_numeric(d.frame.iloc[:, -1], errors="coerce")
    return X, y


def metadados(nome):
    X, y = carregar(nome)
    cat = X.select_dtypes(exclude="number").columns
    return {
        "dataset": nome,
        "linhas": int(X.shape[0]),
        "atributos": int(X.shape[1]),
        "categoricos": int(len(cat)),
        "pct_categoricos": round(100 * len(cat) / X.shape[1], 1),
        "pct_ausentes": round(100 * float(X.isna().mean().mean()), 2),
        "pct_zeros_numericos": round(100 * float((X.select_dtypes("number") == 0).mean().mean()), 2),  # proxy de esparsidade
        "cardinalidade_max_cat": int(max([X[c].nunique() for c in cat], default=0)),
        "alvo_media": round(float(y.mean()), 4),
        "alvo_desvio": round(float(y.std()), 4),
    }


if __name__ == "__main__":
    # ignora argumentos injetados por notebooks (ex.: "-f kernel.json"); só aceita nomes do catálogo
    nomes = [a for a in sys.argv[1:] if a in CANDIDATOS] or list(CANDIDATOS)
    os.makedirs("resultados", exist_ok=True)
    linhas = []
    for n in nomes:
        try:
            linhas.append(metadados(n))
        except Exception as e:
            print(f"[FALHOU] {n}: {e}")
    df = pd.DataFrame(linhas)
    print(df.to_string(index=False))
    df.to_csv("resultados/datasets_meta.csv", index=False)
    with open("resultados/datasets_meta.json", "w", encoding="utf-8") as f:
        json.dump(linhas, f, indent=2, ensure_ascii=False)
