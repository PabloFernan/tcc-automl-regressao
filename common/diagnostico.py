"""Diagnóstico de ambiente: registra versões e CPU/GPU. Rodar no topo de todo notebook."""
import json, os, platform, subprocess, sys
from importlib import metadata

PACOTES = ["autogluon.tabular", "flaml", "h2o", "tpot", "pycaret", "auto-sklearn",
           "scikit-learn", "pandas", "numpy", "scipy", "torch", "lightgbm", "xgboost", "catboost"]


def versoes():
    out = {}
    for p in PACOTES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def gpu():
    info = {"nvidia_smi": None, "torch_cuda": None, "torch_build": None}
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip() or None
    except Exception:
        pass
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        info["torch_build"] = torch.__version__  # "+cpu" => PyTorch sem suporte a GPU
    except Exception:
        pass
    return info


def ambiente():
    import psutil
    g = gpu()
    return {
        "python": sys.version.split()[0],
        "so": platform.platform(),
        "cpus": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "gpu": g,
        # ATENÇÃO: isto diz só que a GPU está DISPONÍVEL. O uso real por ferramenta está em
        # gpu_usada / gpu_usada_mb de cada JSON de resultado (medido durante a execução).
        "gpu_disponivel": bool(g["torch_cuda"]),
        "versoes": versoes(),
    }


def registrar(caminho="resultados/ambiente.json"):
    amb = ambiente()
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(amb, f, indent=2, ensure_ascii=False)
    print(json.dumps(amb, indent=2, ensure_ascii=False))
    return amb


if __name__ == "__main__":
    registrar()
