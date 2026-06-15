import sys
from pathlib import Path
from src.base_model.model_ResNet50 import ResNet50V2_manual

# Carrega o caminho do arquivo e insere o caminho do utils no sys.path.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src" / "training"))

# Carrega as configurações do projeto.
from config import MODELS_DIR, create_model
from train_utils import treinar_geral

# Define o nome do modelo.
NOME_MODELO = "resnet50v2 manual"

if __name__ == "__main__":
    # Treina o modelo.
    treinar_geral(NOME_MODELO, MODELS_DIR, ResNet50V2_manual)
