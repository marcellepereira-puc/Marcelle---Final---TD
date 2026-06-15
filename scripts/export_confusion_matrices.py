"""Exporta matrizes de confusão (CNN ResNet e Siamese) para JSON/LaTeX."""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "utils"))
sys.path.insert(0, str(ROOT / "src" / "training"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

from config import CLASS_NAMES
from dataset_loader import carregar_dados, prever
from evaluate_siamese import avaliar_classificacao
from gpu_config import configurar_gpu
from train_siamese import _carregar_modulo_siamese, _caminho_pesos_embedding

EXPERIMENTOS_CNN = {
    "01": ("covid-chestxray-dataset-master",) * 3,
    "03": ("Imbalanced-Tuberculosis",) * 3,
    "05": ("Imbalanced-Tuberculosis", "covid-chestxray-dataset-master", "makedataset"),
}

SIAMESE_RUNS = [
    ("03_Siamese_imagenet_50epocas", "siamese_resnet50v2_imagenet"),
    ("04_Siamese_imagenet", "siamese_resnet50v2_imagenet"),
]

SIAMESE_EXPS = {
    "01": "covid-chestxray-dataset-master",
    "03": "Imbalanced-Tuberculosis",
    "05": "all",
}


def _cm_dict(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    pct = (cm / total * 100).round(1) if total else cm
    return {
        "classes": CLASS_NAMES,
        "matrix": cm.tolist(),
        "matrix_pct": pct.tolist(),
        "total": total,
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }


def export_cnn(exp: str):
    if exp == "05":
        train_ds, val_ds, test_ds = (
            carregar_dados("train", "Imbalanced-Tuberculosis"),
            carregar_dados("val", "covid-chestxray-dataset-master"),
            carregar_dados("test", "makedataset"),
        )
    else:
        ds = EXPERIMENTOS_CNN[exp][0]
        train_ds, val_ds, test_ds = carregar_dados("train", ds), carregar_dados("val", ds), carregar_dados("test", ds)

    model_path = ROOT / "results" / "02_ResNet50v2" / exp / "resnet50v2.melhor_modelo.pt"
    if not model_path.exists():
        print(f"[skip] CNN {exp}: modelo ausente")
        return None
    model = tf.keras.models.load_model(model_path)
    y_true, y_pred = prever(model, test_ds)
    return _cm_dict(np.array(y_true), np.array(y_pred), list(range(len(CLASS_NAMES))))


def export_siamese(folder: str, modelo: str, exp: str, shot: int):
    pasta = ROOT / "results" / folder / exp / f"{shot}-shot"
    pesos = _caminho_pesos_embedding(modelo, pasta)
    if not pesos.exists():
        print(f"[skip] Siamese {folder} {exp} {shot}-shot: pesos ausentes")
        return None

    modulo = _carregar_modulo_siamese()
    rede = modulo.criar_modelo_embedding(input_shape=(224, 224, 3), dim_embedding=128, freeze_layers=130)
    rede.load_weights(pesos)

    from evaluate_siamese import _coletar_teste, _embedding, _scores_e_predicao, _banco_suporte, _prototipos_de_banco

    banco = _banco_suporte(rede, exp, shot)
    prototipos = _prototipos_de_banco(banco)
    arquivos, y_true = _coletar_teste(SIAMESE_EXPS[exp])
    y_pred = []
    for caminho in arquivos:
        emb = _embedding(rede, caminho)
        pred, _ = _scores_e_predicao(emb, prototipos, banco)
        y_pred.append(pred)
    return _cm_dict(np.array(y_true), np.array(y_pred), list(range(len(CLASS_NAMES))))


def main():
    configurar_gpu()
    out = ROOT / "results" / "confusion_matrices_export.json"
    data = {"cnn_resnet": {}, "siamese": {}}

    for exp in ["01", "03", "05"]:
        cm = export_cnn(exp)
        if cm:
            data["cnn_resnet"][exp] = cm
            print(f"CNN {exp}: ok ({cm['total']} img)")

    for folder, modelo in SIAMESE_RUNS:
        data["siamese"][folder] = {}
        for exp in ["01", "03", "05"]:
            data["siamese"][folder][exp] = {}
            for shot in [1, 3, 5]:
                cm = export_siamese(folder, modelo, exp, shot)
                if cm:
                    data["siamese"][folder][exp][f"{shot}-shot"] = cm
                    print(f"Siamese {folder} {exp} {shot}-shot: ok")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()
