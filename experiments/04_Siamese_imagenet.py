# Experimentos Siamese — ResNet50V2 com pesos ImageNet (01--05)



import argparse

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src" / "utils"))

sys.path.insert(0, str(ROOT / "src" / "training"))



from config import EXPERIMENTS_SIAMESE_IMAGENET_DIR

from train_siamese import rodar_avaliacoes, rodar_experimentos



NOME_MODELO = "siamese_resnet50v2_imagenet"

SHOTS = [1, 3, 5]



EXPERIMENTOS = [

    #("01", "covid-chestxray-dataset-master"),

    #("02", "CXR8"),

    #("03", "Imbalanced-Tuberculosis"),

    ("04", "makedataset"),

    ("05", "all"),

]





if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Siamese ImageNet — treino + teste few-shot nos experimentos 01--05"

    )

    parser.add_argument(

        "--somente-teste",

        action="store_true",

        help="So avalia few-shot no split/test (pesos ja treinados em results/)",

    )

    args = parser.parse_args()



    if args.somente_teste:

        rodar_avaliacoes(

            EXPERIMENTOS,

            NOME_MODELO,

            EXPERIMENTS_SIAMESE_IMAGENET_DIR,

            SHOTS,

            "Siamese ResNet50v2 ImageNet (teste)",

        )

    else:

        rodar_experimentos(

            EXPERIMENTOS,

            NOME_MODELO,

            EXPERIMENTS_SIAMESE_IMAGENET_DIR,

            SHOTS,

            "Siamese ResNet50v2 ImageNet",

        )

