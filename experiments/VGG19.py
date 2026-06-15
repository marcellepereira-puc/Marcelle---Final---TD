# Experimentos VGG19 — usa somente datasets/split

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src" / "training"))

from config import EXPERIMENTS_VGG19_DIR, create_vgg19_model
from train_utils import rodar_experimentos

NOME_MODELO = "vgg19"

EXPERIMENTOS = [
    ("01", "covid-chestxray-dataset-master", "covid-chestxray-dataset-master", "covid-chestxray-dataset-master"),
    ("02", "CXR8", "CXR8", "CXR8"),
    ("03", "Imbalanced-Tuberculosis", "Imbalanced-Tuberculosis", "Imbalanced-Tuberculosis"),
    ("04", "makedataset", "makedataset", "makedataset"),
    ("05", "Imbalanced-Tuberculosis", "covid-chestxray-dataset-master", "makedataset"),
]

if __name__ == "__main__":
    rodar_experimentos(EXPERIMENTOS, NOME_MODELO, EXPERIMENTS_VGG19_DIR, create_vgg19_model, "VGG19")
