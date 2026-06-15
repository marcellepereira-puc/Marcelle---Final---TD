# Executar treino Siamese — usa .venv (Python 3.10 + TensorFlow + DirectML)

import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "2")

ROOT = Path(__file__).resolve().parent.parent
PAIR_DIR = ROOT / "datasets" / "pair"
SPLIT_DIR = ROOT / "datasets" / "split"

PASSOS = [
    ("Gerar pairs", ROOT / "src" / "data" / "ger_pair.py"),
    ("Treinar Siamese", ROOT / "experiments" / "Siamese.py"),
]


def _python():
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv)
    raise SystemExit(
        "Ambiente .venv não encontrado.\n"
        "Execute:\n"
        "  py -3.10 -m venv .venv\n"
        "  .venv\\Scripts\\activate\n"
        "  pip install -r requirements.txt"
    )


def _tensorflow_ok(python_exe: str) -> bool:
    resultado = subprocess.run(
        [
            python_exe,
            "-c",
            "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; "
            "import tensorflow as tf; "
            "assert tf.config.list_physical_devices('GPU'); "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
    )
    return resultado.returncode == 0


def _formatar_tempo(segundos: float) -> str:
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _pair_pronto() -> bool:
    """Verifica se existe pelo menos um experimento com ancora/positiva/negativa."""
    if not PAIR_DIR.exists():
        return False

    for pasta in PAIR_DIR.iterdir():
        if not pasta.is_dir():
            continue
        for shot in ("1-shot", "3-shot", "5-shot"):
            pastas = (
                pasta / "ancora" / shot,
                pasta / "positiva" / shot,
                pasta / "negativa" / shot,
            )
            if all(p.is_dir() for p in pastas):
                return True
    return False


def main():
    py = _python()

    if not _tensorflow_ok(py):
        raise SystemExit(
            f"TensorFlow/GPU indisponível em {py}\n"
            "Execute: .venv\\Scripts\\activate && pip install -r requirements.txt"
        )

    print(f"Python: {py}\n")

    if not SPLIT_DIR.exists():
        print(f"[aviso] Split não encontrado: {SPLIT_DIR}")
        print("Execute: python src/data/ger_split.py\n")
        return

    precisa_gerar_pair = not _pair_pronto()
    if precisa_gerar_pair:
        print(f"[aviso] Pairs não encontrados em: {PAIR_DIR}")
        print("Será executado ger_pair.py antes do treino.\n")
    else:
        print(f"Pairs encontrados em: {PAIR_DIR}\n")

    pulados, falhas, tempos = [], [], {}
    inicio_total = time.time()

    for nome, script in PASSOS:
        if nome == "Gerar pairs" and not precisa_gerar_pair:
            print(f"--- {nome}: já existe, pulando ---\n")
            continue

        if not script.exists():
            pulados.append(script.name)
            continue

        print(f"--- {nome}: {script.relative_to(ROOT)} ---")
        t0 = time.time()
        try:
            subprocess.run([py, str(script)], check=True, cwd=str(ROOT))
            tempos[nome] = _formatar_tempo(time.time() - t0)
        except subprocess.CalledProcessError:
            falhas.append(nome)
            print(f"[aviso] Falha em {nome}, pulando...\n")
            if nome == "Gerar pairs":
                print("Treino Siamese cancelado — pairs não gerados.\n")
                break

    print(f"\nTempo total: {_formatar_tempo(time.time() - inicio_total)}")
    for nome, t in tempos.items():
        print(f"  {nome}: {t}")
    if pulados:
        print("Pulados:", ", ".join(pulados))
    if falhas:
        print("Falhas:", ", ".join(falhas))
    elif tempos:
        print("\nConcluído.")


if __name__ == "__main__":
    main()
