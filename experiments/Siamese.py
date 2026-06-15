# Experimentos Siamese — usa datasets/pair

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "training"))

from config import EXPERIMENTS_SIAMESE_DIR
from train_siamese import rodar_experimentos

NOME_MODELO = "siamese_resnet50v2"
SHOTS = [1, 3, 5]

EXPERIMENTOS = [
    ("01", "covid-chestxray-dataset-master"),
    ("02", "CXR8"),
    ("03", "Imbalanced-Tuberculosis"),
    ("04", "makedataset"),
    ("05", "all"),
]

if __name__ == "__main__":
    rodar_experimentos(
        EXPERIMENTOS,
        NOME_MODELO,
        EXPERIMENTS_SIAMESE_DIR,
        SHOTS,
        "Siamese ResNet50v2",
    )
