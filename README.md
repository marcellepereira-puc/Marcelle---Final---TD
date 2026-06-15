# Trabalho de Diplomação

**Aluna:** Marcelle Andrade Pereira  
**Instituição:** Pontifícia Universidade Católica de Minas Gerais — Campus Poços de Caldas  
**Orientador:** Professor Leonardo de Melo João

---

## Sobre o projeto

Este repositório contém o código, os experimentos e os resultados do Trabalho de Diplomação que compara abordagens de **aprendizado profundo para classificação de radiografias de tórax**. O estudo abrange quatro classes clínicas:

| Classe | Descrição |
|--------|-----------|
| **No Finding** | Sem achados |
| **Pneumonia** | Pneumonia |
| **Tuberculosis** | Tuberculose |
| **Convid** | COVID-19 |

Foram implementados e avaliados três tipos de modelos:

1. **VGG19** — rede convolucional clássica treinada do zero para classificação multiclasse.
2. **ResNet50V2** — rede residual pré-treinada adaptada para classificação multiclasse.
3. **Rede Siamese (ResNet50V2 + ImageNet)** — aprendizado métrico com *triplet loss*, avaliado em cenários *few-shot* (1, 3 e 5 exemplos por classe).

Os experimentos utilizam cinco configurações de dados (01–05), combinando diferentes bases de radiografias de tórax disponíveis publicamente.

---

## Datasets

| ID | Nome |
|----|------|
| 01 | `covid-chestxray-dataset-master` |
| 02 | `CXR8` |
| 03 | `Imbalanced-Tuberculosis` |
| 04 | `makedataset` |
| 05 | `all` (combinação de múltiplas bases) |

Os dados devem ser organizados em `datasets/` conforme a estrutura descrita abaixo. Os scripts em `src/data/` automatizam o pré-processamento, a divisão e a geração de tripletas.

---

## Estrutura do repositório

```
├── datasets/
│   ├── processed/          # Imagens pré-processadas (v1, v2)
│   ├── split/              # Divisão train / val / test por experimento
│   └── pair/               # Tripletas para treino Siamese (âncora, positiva, negativa)
│
├── src/
│   ├── base_model/         # Arquiteturas (VGG19, ResNet50V2, Siamese)
│   ├── data/               # Pré-processamento, splits e contagem de dados
│   ├── evaluation/         # Métricas, matriz de confusão, t-SNE, ROC
│   ├── training/           # Treino e carregamento de datasets
│   └── utils/              # Configuração de GPU, transforms e utilitários
│
├── experiments/            # Pontos de entrada para rodar os experimentos
│   ├── VGG19.py
│   ├── ResNet50v2.py
│   ├── Siamese.py
│   └── 04_Siamese_imagenet.py
│
├── results/                # Modelos treinados, métricas e gráficos
│   ├── 01_VGG_19/
│   ├── 02_ResNet50v2/
│   ├── 03_Siamese_imagenet_50epocas/
│   └── 04_Siamese_imagenet/
│
└── requirements.txt
```

---

## Requisitos

- **Python 3.10**
- **GPU** (CUDA ou DirectML no Windows) — o treino exige aceleração por GPU
- Dependências listadas em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Principais bibliotecas: TensorFlow 2.10, NumPy, Pandas, Pillow, Matplotlib e scikit-learn.

---

## Preparação dos dados

Execute os scripts em `src/data/` na ordem abaixo (a partir da raiz do repositório):

```bash
# 1. Pré-processar imagens (redimensionamento, normalização)
python src/data/ger_processed.py

# 2. Gerar splits train / val / test (70% / 15% / 15%)
python src/data/ger_split.py

# 3. Gerar splits combinados (experimento 05 — "all")
python src/data/ger_split_all.py
```

Scripts auxiliares de contagem: `cont_img.py` e `cont_pair.py` (gera relatório CSV das tripletas em `datasets/pair/`).

---

## Executando os experimentos

Todos os scripts de experimento devem ser executados a partir da raiz do repositório.

### VGG19

```bash
python experiments/VGG19.py
```

### ResNet50V2

```bash
python experiments/ResNet50v2.py
```

### Siamese (ResNet50V2 + ImageNet, few-shot)

```bash
# Treino completo + avaliação few-shot
python experiments/04_Siamese_imagenet.py

# Apenas avaliação (usa pesos já salvos em results/)
python experiments/04_Siamese_imagenet.py --somente-teste
```

Cada experimento (01–05) gera, em `results/`, os seguintes artefatos:

- Modelo com melhor `val_loss`
- Gráficos de treino, matriz de confusão e curva ROC
- Métricas (accuracy, precision, recall, F1) em JSON
- Tempo de treino em JSON
- Visualização t-SNE dos embeddings (Siamese)

Documentação detalhada com **todas as figuras comentadas**: [`doc/analise_resultados.md`](doc/analise_resultados.md).

### Exemplos visuais (melhores resultados)

| VGG19 (96,96%) | ResNet50V2 (97,15%) | Siamese ImageNet (99,67%) |
|:---:|:---:|:---:|
| ![VGG19](../results/01_VGG_19/02/vgg19_confusion_matrix.png) | ![ResNet](../results/02_ResNet50v2/03/resnet50v2_confusion_matrix.png) | ![Siamese](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png) |
| Exp. 02 — CXR8 | Exp. 03 — Imbalanced TB | Exp. 04 — 5-shot |
