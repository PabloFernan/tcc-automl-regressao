"""Runner único: mesma interface, split, budget e métricas para todas as ferramentas.

Uso (Colab):  from runner import executar_tudo; executar_tudo(["flaml","autogluon"], ["abalone"], seeds=range(42,47))
Cada execução grava resultados/{tool}__{dataset}__seed{n}.json; execuções já feitas são puladas (retomável).
"""
import json, logging, os, sys, threading, time
import numpy as np

# silencia o ruído do Dask/FLAML (mensagens INFO em vermelho no Colab)
for _n in ("distributed", "flaml", "tornado"):
    logging.getLogger(_n).setLevel(logging.ERROR)
import psutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from datasets_meta import carregar
from diagnostico import ambiente

BUDGET_S = 300
# Ajustes do TPOT ao budget de 300 s, os MESMOS das seeds 42-43 (defaults do TPOT estouravam o tempo).
# Fixados aqui para não depender de lembrar de configurar a cada sessão do Colab; não alterar no meio do estudo.
TPOT_PARAMS = {"cv": 3, "max_eval_time_mins": 1}
SEEDS = range(42, 47)
DIR = "resultados"


class PicoRAM:
    """Amostra RSS do processo + filhos (AutoGluon/H2O usam subprocessos/JVM) e guarda o pico."""
    def __enter__(self):
        self.base = psutil.Process().memory_info().rss  # memória já ocupada antes da execução
        self.gpu_base = self._gpu_mb()  # GPU já ocupada antes da execução (None = sem GPU)
        self.gpu_pico = None
        self.pico, self._run = 0, True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    @staticmethod
    def _gpu_mb():
        """Memória da GPU em uso (MB) via nvidia-smi; None se não houver GPU."""
        try:
            import subprocess
            s = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            return float(s.splitlines()[0])
        except Exception:
            return None

    def _loop(self):
        p = psutil.Process()
        i = 0
        while self._run:
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                self.pico = max(self.pico, rss)
            except Exception:
                pass
            if i % 4 == 0:  # a cada ~2 s
                g = self._gpu_mb()
                if g is not None:
                    self.gpu_pico = max(self.gpu_pico or 0, g)
            i += 1
            time.sleep(0.5)

    def __exit__(self, *a):
        self._run = False
        self._t.join()

    @property
    def mb(self):
        return round(self.pico / 2**20, 1)

    @property
    def gpu_delta_mb(self):
        """Aumento de memória da GPU durante a execução. ~0 => a ferramenta não usou a GPU."""
        if self.gpu_base is None or self.gpu_pico is None:
            return None
        return round(max(self.gpu_pico - self.gpu_base, 0), 1)

    @property
    def delta_mb(self):  # pico menos o que já estava carregado (Python, torch, dados)
        return round(max(self.pico - self.base, 0) / 2**20, 1)


# ---- adaptadores: cada um recebe (Xtr, ytr, Xte, seed, budget) e devolve (preds, versao) ----
def _flaml(Xtr, ytr, Xte, seed, budget):
    import flaml
    from flaml import AutoML
    m = AutoML()
    m.fit(Xtr, ytr, task="regression", metric="rmse", time_budget=budget, seed=seed, verbose=0)
    return m.predict(Xte), flaml.__version__


def _autogluon(Xtr, ytr, Xte, seed, budget):
    import pandas as pd, shutil, tempfile
    from autogluon.tabular import TabularPredictor
    import autogluon.tabular as agt
    d = tempfile.mkdtemp()
    tr = Xtr.copy(); tr["__y__"] = ytr.values
    p = TabularPredictor(label="__y__", problem_type="regression", eval_metric="root_mean_squared_error",
                         path=d, verbosity=0)
    # ag_args_fit.random_seed: seed do AutoGluon (o script antigo não passava seed)
    p.fit(tr, time_limit=budget, presets="high_quality", ag_args_fit={"random_seed": seed})
    preds = p.predict(Xte).values
    shutil.rmtree(d, ignore_errors=True)
    return preds, agt.__version__


def _h2o(Xtr, ytr, Xte, seed, budget):
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init(verbose=False)
    tr = Xtr.copy(); tr["__y__"] = ytr.values
    htr, hte = h2o.H2OFrame(tr), h2o.H2OFrame(Xte)
    a = H2OAutoML(max_runtime_secs=budget, seed=seed, sort_metric="RMSE")
    a.train(y="__y__", training_frame=htr)
    preds = a.leader.predict(hte).as_data_frame().values.ravel()
    v = h2o.__version__
    h2o.cluster().shutdown(prompt=False)
    return preds, v


def _tpot(Xtr, ytr, Xte, seed, budget):
    import pandas as pd, tpot
    from tpot import TPOTRegressor
    # TPOT só aceita numérico: one-hot nas categóricas (mesma informação que as outras ferramentas veem)
    n = len(Xtr)
    todo = pd.get_dummies(pd.concat([Xtr, Xte]), dtype=float)
    todo = todo.fillna(todo.iloc[:n].median())
    Ftr, Fte = todo.iloc[:n], todo.iloc[n:]
    import inspect
    # Parâmetros adaptados ao budget curto (defaults do TPOT 1.x: cv=10, avaliação de até 10 min por
    # pipeline e população de 50 -> estoura o budget e quase não completa gerações). Ajustar aqui.
    # n_jobs=-1 falhou no Colab (Dask sem workers); testar n_jobs=2 (2 vCPU) em TPOT_PARAMS.
    params = dict(max_time_mins=budget / 60, random_state=seed, verbose=0, n_jobs=1,
                  scorers=["neg_root_mean_squared_error"], scorers_weights=[1])
    params.update(TPOT_PARAMS)
    aceitos = inspect.signature(TPOTRegressor.__init__).parameters
    ignorados = [k for k in params if k not in aceitos]
    if ignorados:
        print(f"[TPOT] parâmetros inexistentes nesta versão (ignorados): {ignorados}")
    m = TPOTRegressor(**{k: v for k, v in params.items() if k in aceitos})
    m.fit(Ftr, ytr)
    return m.predict(Fte), tpot.__version__


FERRAMENTAS = {"flaml": _flaml, "autogluon": _autogluon, "h2o": _h2o, "tpot": _tpot}


def executar(tool, dataset, seed, budget=BUDGET_S, dir_=DIR):
    os.makedirs(dir_, exist_ok=True)
    caminho = f"{dir_}/{tool}__{dataset}__seed{seed}.json"
    if os.path.exists(caminho):
        return json.load(open(caminho, encoding="utf-8"))
    X, y = carregar(dataset)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed)
    amb = ambiente()
    t0 = time.time()
    with PicoRAM() as ram:
        preds, versao = FERRAMENTAS[tool](Xtr, ytr, Xte, seed, budget)
    res = {
        "framework": tool, "versao": versao, "dataset": dataset, "seed": seed, "budget_s": budget,
        "tempo_real_s": round(time.time() - t0, 2), "ram_pico_mb": ram.mb, "ram_delta_mb": ram.delta_mb,
        "rmse": float(np.sqrt(mean_squared_error(yte, preds))),
        "mae": float(mean_absolute_error(yte, preds)), "r2": float(r2_score(yte, preds)),
        # gpu_disponivel != gpu_usada: a T4 pode estar alocada sem a ferramenta usá-la.
        # gpu_usada_mb = aumento da memória da GPU durante a execução (medido); ~0 => rodou em CPU.
        "gpu_disponivel": bool(amb["gpu"]["torch_cuda"]), "gpu_usada_mb": ram.gpu_delta_mb,
        "gpu_usada": (ram.gpu_delta_mb or 0) > 50,
        "ambiente": {k: amb[k] for k in ("python", "cpus", "ram_total_gb", "gpu")},
    }
    # previsões guardadas p/ diagnosticar outliers depois (não afeta a consolidação, que lê só *.json)
    np.savez_compressed(caminho.replace(".json", "_pred.npz"), y=np.asarray(yte), p=np.asarray(preds))
    json.dump(res, open(caminho, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"OK {tool} {dataset} seed{seed}: RMSE={res['rmse']:.4f} t={res['tempo_real_s']}s RAM pico={res['ram_pico_mb']}MB (+{res['ram_delta_mb']}MB)")
    return res


def _gravar_falha(dir_, tool, dataset, seed, budget, t0, erro):
    os.makedirs(dir_, exist_ok=True)
    json.dump({"framework": tool, "dataset": dataset, "seed": seed, "budget_s": budget,
               "tempo_ate_falhar_s": round(time.time() - t0, 1), "erro": str(erro)[-600:]},
              open(f"{dir_}/{tool}__{dataset}__seed{seed}.falha", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def _executar_ou_falhar(tool, dataset, seed, budget, dir_):
    """Roda DENTRO do subprocesso: executa e, se der erro, registra o .falha e sai com código 1."""
    t0 = time.time()
    try:
        executar(tool, dataset, seed, budget, dir_)
    except Exception as e:
        print(f"[FALHOU] {tool} {dataset} seed{seed}: {e}")
        _gravar_falha(dir_, tool, dataset, seed, budget, t0, e)
        sys.exit(1)


def _rodar_isolado(tool, dataset, seed, budget, dir_, timeout_s):
    """Cada execução em um processo novo: a memória volta ao zero a cada vez (no mesmo processo ela se acumulava
    ~4 GB por semente e derrubava a sessão do Colab) e um limite rígido de tempo mata execuções travadas."""
    import signal, subprocess
    aqui = os.path.dirname(os.path.abspath(__file__))
    code = (f"import sys; sys.path.insert(0, {aqui!r}); import runner; "
            f"runner._executar_ou_falhar({tool!r}, {dataset!r}, {seed}, {budget}, {dir_!r})")
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        saida = p.communicate(timeout=timeout_s)[0]
        motivo = f"processo terminou com código {p.returncode} sem gravar resultado"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)  # mata também a JVM do H2O
        except Exception:
            p.kill()
        saida = p.communicate()[0]
        motivo = f"excedeu o limite rígido de {timeout_s} s e foi encerrado"
    linhas = [l for l in (saida or "").splitlines() if l.strip()]
    print(chr(10).join(linhas[-6:]))
    ok = os.path.exists(f"{dir_}/{tool}__{dataset}__seed{seed}.json")
    if not ok and not os.path.exists(f"{dir_}/{tool}__{dataset}__seed{seed}.falha"):
        _gravar_falha(dir_, tool, dataset, seed, budget, t0, motivo + " | " + " ".join(linhas[-3:]))
        print(f"[FALHOU] {tool} {dataset} seed{seed}: {motivo}")


def executar_tudo(tools, datasets, seeds=SEEDS, budget=BUDGET_S, dir_=DIR, retentar_falhas=False, isolar=True, timeout_s=1200):
    # seed é o laço externo: se o Colab cair, já existe comparação completa com menos seeds
    total = len(tools) * len(datasets) * len(seeds)
    feitas = sum(os.path.exists(f"{dir_}/{t}__{d}__seed{s}.json") for t in tools for d in datasets for s in seeds)
    falhas = sum(os.path.exists(f"{dir_}/{t}__{d}__seed{s}.falha") for t in tools for d in datasets for s in seeds)
    print(f"{feitas}/{total} execuções já concluídas, {falhas} falhas registradas (não serão repetidas)")
    for s in seeds:
        for d in datasets:
            for t in tools:
                if os.path.exists(f"{dir_}/{t}__{d}__seed{s}.json"):
                    continue
                falha = f"{dir_}/{t}__{d}__seed{s}.falha"
                if os.path.exists(falha):
                    if not retentar_falhas:
                        continue  # falha já registrada: segue o protocolo (não reexecuta), como as 2 falhas do TPOT na rodada 1
                    os.remove(falha)
                if isolar:
                    _rodar_isolado(t, d, s, budget, dir_, timeout_s)
                    continue
                t0 = time.time()
                try:
                    executar(t, d, s, budget, dir_)
                except Exception as e:  # uma falha não derruba o lote
                    print(f"[FALHOU] {t} {d} seed{s}: {e}")
                    _gravar_falha(dir_, t, d, s, budget, t0, e)
