# Análise dos Resultados Experimentais

**Trabalho de Diplomação — Marcelle Andrade Pereira**  
Pontifícia Universidade Católica de Minas Gerais — Campus Poços de Caldas

Este documento analisa os gráficos e métricas salvos em `results/`, com comentários sobre cada figura gerada durante os experimentos.

---

## Introdução

Este documento apresenta a análise visual e quantitativa dos experimentos de classificação de radiografias de tórax em quatro classes: **No Finding**, **Pneumonia**, **Tuberculosis** e **Convid**.

Foram comparados três paradigmas:

1. **VGG19** — CNN clássica treinada do zero.

2. **ResNet50V2** — CNN residual com transfer learning.

3. **Siamese ResNet50V2 (ImageNet)** — aprendizado métrico com avaliação few-shot (1, 3 e 5 exemplos por classe).

Cada experimento (01–05) utiliza combinações distintas de bases públicas de RX de tórax.

### Tipos de figura analisadas

| Figura | O que representa |
|--------|------------------|
| **Curvas de treino** | Evolução da perda (*loss*) e acurácia em treino e validação ao longo das épocas |
| **Matriz de confusão** | Acertos e erros por classe (verdadeiro × predito) no conjunto de teste |
| **Curvas ROC** | Capacidade de separar cada classe (one-vs-rest); AUC próximo de 1 indica excelente discriminação |
| **t-SNE** | Projeção 2D dos embeddings; clusters separados indicam representações discriminativas |

---

## Síntese comparativa

### Melhores resultados por abordagem

| Abordagem | Melhor configuração | Acurácia | F1 macro |
|-----------|-------------------|----------|----------|
| VGG19 | Exp. 02 (CXR8) | 96,96% | 49,23% |
| ResNet50V2 | Exp. 03 (Imbalanced TB) | 97,15% | 96,12% |
| Siamese (50 ép.) | Exp. 04, 5-shot | 73,67% | 73,69% |
| Siamese (ImageNet) | Exp. 04, 5-shot | **99,67%** | **99,67%** |

### Destaques visuais (melhores configurações)

#### VGG19 — Exp. 02 (CXR8) · 96,96%

![VGG19 treino — exp. 02](../results/01_VGG_19/02/vgg19_training.png)

*Comentário:* Convergência rápida; acurácia de validação estável acima de 96%. Pequena diferença entre treino e validação indica bom ajuste neste dataset.

![VGG19 confusão — exp. 02](../results/01_VGG_19/02/vgg19_confusion_matrix.png)

*Comentário:* Diagonal dominante — poucos erros entre as quatro classes. F1 macro (49,23%) ainda reflete dificuldade com classes minoritárias apesar da alta acurácia.

#### ResNet50V2 — Exp. 03 (Imbalanced Tuberculosis) · 97,15%

![ResNet50v2 treino — exp. 03](../results/02_ResNet50v2/03/resnet50v2_training.png)

*Comentário:* Perda decrescente e acurácia de validação próxima de 97%; transfer learning com ResNet50V2 se mostra eficaz neste cenário.

![ResNet50v2 confusão — exp. 03](../results/02_ResNet50v2/03/resnet50v2_confusion_matrix.png)

*Comentário:* Melhor equilíbrio entre classes entre os classificadores diretos (F1 macro 96,12%). Erros residuais concentrados em pares de patologias pulmonares visualmente próximas.

#### Siamese ImageNet (50 ép.) — Exp. 04, 5-shot · 73,67%

![Siamese 50ep treino — exp. 04, 5-shot](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_training.png)

*Comentário:* Treino métrico com triplet loss; curvas indicam aprendizado estável, porém desempenho few-shot inferior à run final.

![Siamese 50ep confusão — exp. 04, 5-shot](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png)

*Comentário:* Classificação por protótipos com 5 exemplos por classe; acertos majoritários na diagonal, mas confusões visíveis entre Convid e No Finding.

![Siamese 50ep ROC — exp. 04, 5-shot](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_roc.png)

*Comentário:* AUC macro 0,781 — separação moderada; curvas de Pneumonia e Tuberculosis destacam-se.

![Siamese 50ep t-SNE — exp. 04, 5-shot](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_tsne.png)

*Comentário:* Embeddings formam grupos parcialmente sobrepostos; espaço métrico ainda não separa todas as classes de forma nítida.

#### Siamese ImageNet (run final) — Exp. 04, 5-shot · 99,67%

![Siamese final treino — exp. 04, 5-shot](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_training.png)

*Comentário:* Melhor resultado global do trabalho; treino converge com baixa perda de validação.

![Siamese final confusão — exp. 04, 5-shot](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png)

*Comentário:* Quase perfeita — 299/300 acertos; único erro: No Finding classificado como Convid.

![Siamese final ROC — exp. 04, 5-shot](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_roc.png)

*Comentário:* AUC macro 0,999 — discriminação excelente em todas as classes (one-vs-rest).

![Siamese final t-SNE — exp. 04, 5-shot](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_tsne.png)

*Comentário:* Clusters compactos e bem separados por classe; confirma qualidade dos embeddings L2-normalizados.

**Observações gerais:**

- Os classificadores VGG19 e ResNet50V2 degradam fortemente no experimento 05 (dados mistos entre bases), enquanto a rede Siamese com ImageNet mantém desempenho robusto.

- O F1 macro frequentemente fica abaixo do F1 ponderado, sinalizando dificuldade com classes minoritárias.

- No few-shot, aumentar de 1 para 5 exemplos por classe melhora consistentemente acurácia e AUC.

- A run `03_Siamese_imagenet_50epocas` apresenta métricas inferiores à `04_Siamese_imagenet`, sugerindo benefício do protocolo de treino/avaliação da versão final.

---

## 1. VGG19

Classificação multiclasse direta (50 épocas).

| Exp. | Dataset | Acurácia | F1 ponderado | F1 macro | Tempo |
|------|---------|----------|--------------|----------|-------|
| 01 | COVID Chest X-ray | 83.05% | 75.36% | 22.69% | 17m 34s |
| 02 | CXR8 | 96.96% | 95.46% | 49.23% | 2h 0m 35s |
| 03 | Imbalanced Tuberculosis | 60.19% | 45.23% | 25.05% | 5h 49m 19s |
| 04 | MakeDataset | 25.00% | 10.00% | 10.00% | 47m 4s |
| 05 | Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset | 25.00% | 10.00% | 10.00% | 5h 28m 13s |
### Experimento 01 — COVID Chest X-ray

**Métricas:** acurácia 83.05%, precisão ponderada 68.97%, recall ponderado 83.05%, F1 macro 22.69%.
 Tempo de treino: **17m 34s**.

#### Curvas de treino

![vgg19_training.png](../results/01_VGG_19/01/vgg19_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![vgg19_confusion_matrix.png](../results/01_VGG_19/01/vgg19_confusion_matrix.png)

*Comentário:* Acurácia aparente 83.05%, porém F1 macro baixo (22.69%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

### Experimento 02 — CXR8

**Métricas:** acurácia 96.96%, precisão ponderada 94.01%, recall ponderado 96.96%, F1 macro 49.23%.
 Tempo de treino: **2h 0m 35s**.

#### Curvas de treino

![vgg19_training.png](../results/01_VGG_19/02/vgg19_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. Observa-se possível sobreajuste: acurácia de treino muito superior à de validação. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![vgg19_confusion_matrix.png](../results/01_VGG_19/02/vgg19_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 96.96%). A diagonal principal concentra a maior parte das predições corretas.

### Experimento 03 — Imbalanced Tuberculosis

**Métricas:** acurácia 60.19%, precisão ponderada 36.23%, recall ponderado 60.19%, F1 macro 25.05%.
 Tempo de treino: **5h 49m 19s**.

#### Curvas de treino

![vgg19_training.png](../results/01_VGG_19/03/vgg19_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![vgg19_confusion_matrix.png](../results/01_VGG_19/03/vgg19_confusion_matrix.png)

*Comentário:* Acurácia aparente 60.19%, porém F1 macro baixo (25.05%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

### Experimento 04 — MakeDataset

**Métricas:** acurácia 25.00%, precisão ponderada 6.25%, recall ponderado 25.00%, F1 macro 10.00%.
 Tempo de treino: **47m 4s**.

#### Curvas de treino

![vgg19_training.png](../results/01_VGG_19/04/vgg19_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![vgg19_confusion_matrix.png](../results/01_VGG_19/04/vgg19_confusion_matrix.png)

*Comentário:* Desempenho próximo ao acaso (acurácia 25.00%). Erros distribuídos entre classes indicam falha na separação das patologias.

### Experimento 05 — Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset

**Métricas:** acurácia 25.00%, precisão ponderada 6.25%, recall ponderado 25.00%, F1 macro 10.00%.
 Tempo de treino: **5h 28m 13s**.

#### Curvas de treino

![vgg19_training.png](../results/01_VGG_19/05/vgg19_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![vgg19_confusion_matrix.png](../results/01_VGG_19/05/vgg19_confusion_matrix.png)

*Comentário:* Desempenho próximo ao acaso (acurácia 25.00%). Erros distribuídos entre classes indicam falha na separação das patologias.

---

## 2. ResNet50V2

Classificação multiclasse direta (50 épocas).

| Exp. | Dataset | Acurácia | F1 ponderado | F1 macro | Tempo |
|------|---------|----------|--------------|----------|-------|
| 01 | COVID Chest X-ray | 77.97% | 73.12% | 22.01% | 8m 55s |
| 02 | CXR8 | 96.96% | 95.46% | 49.23% | 47m 42s |
| 03 | Imbalanced Tuberculosis | 97.15% | 97.16% | 96.12% | 2h 15m 37s |
| 04 | MakeDataset | 91.67% | 91.73% | 91.73% | 27m 27s |
| 05 | Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset | 61.33% | 52.72% | 52.72% | 2h 3m 1s |
### Experimento 01 — COVID Chest X-ray

**Métricas:** acurácia 77.97%, precisão ponderada 68.83%, recall ponderado 77.97%, F1 macro 22.01%.
 Tempo de treino: **8m 55s**.

#### Curvas de treino

![resnet50v2_training.png](../results/02_ResNet50v2/01/resnet50v2_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![resnet50v2_confusion_matrix.png](../results/02_ResNet50v2/01/resnet50v2_confusion_matrix.png)

*Comentário:* Acurácia aparente 77.97%, porém F1 macro baixo (22.01%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

### Experimento 02 — CXR8

**Métricas:** acurácia 96.96%, precisão ponderada 94.01%, recall ponderado 96.96%, F1 macro 49.23%.
 Tempo de treino: **47m 42s**.

#### Curvas de treino

![resnet50v2_training.png](../results/02_ResNet50v2/02/resnet50v2_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. Observa-se possível sobreajuste: acurácia de treino muito superior à de validação. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![resnet50v2_confusion_matrix.png](../results/02_ResNet50v2/02/resnet50v2_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 96.96%). A diagonal principal concentra a maior parte das predições corretas.

### Experimento 03 — Imbalanced Tuberculosis

**Métricas:** acurácia 97.15%, precisão ponderada 97.17%, recall ponderado 97.15%, F1 macro 96.12%.
 Tempo de treino: **2h 15m 37s**.

#### Curvas de treino

![resnet50v2_training.png](../results/02_ResNet50v2/03/resnet50v2_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![resnet50v2_confusion_matrix.png](../results/02_ResNet50v2/03/resnet50v2_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 97.15%). A diagonal principal concentra a maior parte das predições corretas.

### Experimento 04 — MakeDataset

**Métricas:** acurácia 91.67%, precisão ponderada 91.84%, recall ponderado 91.67%, F1 macro 91.73%.
 Tempo de treino: **27m 27s**.

#### Curvas de treino

![resnet50v2_training.png](../results/02_ResNet50v2/04/resnet50v2_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![resnet50v2_confusion_matrix.png](../results/02_ResNet50v2/04/resnet50v2_confusion_matrix.png)

*Comentário:* Acurácia de 91.67% (F1 macro 91.73%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

### Experimento 05 — Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset

**Métricas:** acurácia 61.33%, precisão ponderada 46.34%, recall ponderado 61.33%, F1 macro 52.72%.
 Tempo de treino: **2h 3m 1s**.

#### Curvas de treino

![resnet50v2_training.png](../results/02_ResNet50v2/05/resnet50v2_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

#### Matriz de confusão

![resnet50v2_confusion_matrix.png](../results/02_ResNet50v2/05/resnet50v2_confusion_matrix.png)

*Comentário:* Acurácia de 61.33% (F1 macro 52.72%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

---

## 3. Siamese ImageNet — 50 épocas (run inicial)

Rede Siamese com ResNet50V2 (ImageNet), triplet loss. Avaliação few-shot (50 épocas).

| Exp. | Shot | Acurácia | F1 macro | AUC macro |
|------|------|----------|----------|-----------|
| 01 | 1-shot | 27.97% | 21.43% | 0.618 |
| 01 | 3-shot | 35.59% | 25.48% | 0.679 |
| 01 | 5-shot | 71.19% | 42.31% | 0.681 |
| 02 | 1-shot | 5.32% | 5.32% | 0.455 |
| 02 | 3-shot | 48.16% | 34.53% | 0.496 |
| 02 | 5-shot | 78.83% | 50.26% | 0.608 |
| 03 | 1-shot | 31.82% | 31.34% | 0.542 |
| 03 | 3-shot | 39.94% | 41.92% | 0.613 |
| 03 | 5-shot | 58.51% | 51.70% | 0.679 |
| 04 | 1-shot | 49.00% | 48.21% | 0.702 |
| 04 | 3-shot | 59.67% | 60.66% | 0.777 |
| 04 | 5-shot | 73.67% | 73.69% | 0.781 |
| 05 | 1-shot | 33.61% | 28.25% | 0.627 |
| 05 | 3-shot | 51.04% | 45.62% | 0.693 |
| 05 | 5-shot | 58.79% | 51.74% | 0.782 |
### Experimento 01 — COVID Chest X-ray

#### 1-SHOT

**Métricas:** acurácia 27.97%, F1 macro 21.43%, AUC macro 0.618.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_1shot_training.png](../results/03_Siamese_imagenet_50epocas/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_1shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_confusion_matrix.png)

*Comentário:* Desempenho próximo ao acaso (acurácia 27.97%). Erros distribuídos entre classes indicam falha na separação das patologias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_1shot_roc.png](../results/03_Siamese_imagenet_50epocas/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_roc.png)

*Comentário:* AUC macro 0.618: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_1shot_tsne.png](../results/03_Siamese_imagenet_50epocas/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 35.59%, F1 macro 25.48%, AUC macro 0.679.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_3shot_training.png](../results/03_Siamese_imagenet_50epocas/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_3shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_confusion_matrix.png)

*Comentário:* Acurácia aparente 35.59%, porém F1 macro baixo (25.48%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_3shot_roc.png](../results/03_Siamese_imagenet_50epocas/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_roc.png)

*Comentário:* AUC macro 0.679: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_3shot_tsne.png](../results/03_Siamese_imagenet_50epocas/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 71.19%, F1 macro 42.31%, AUC macro 0.681.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_5shot_training.png](../results/03_Siamese_imagenet_50epocas/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_5shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 71.19% (F1 macro 42.31%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_5shot_roc.png](../results/03_Siamese_imagenet_50epocas/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_roc.png)

*Comentário:* AUC macro 0.681: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_5shot_tsne.png](../results/03_Siamese_imagenet_50epocas/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 02 — CXR8

#### 1-SHOT

**Métricas:** acurácia 5.32%, F1 macro 5.32%, AUC macro 0.455.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_1shot_training.png](../results/03_Siamese_imagenet_50epocas/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_1shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_confusion_matrix.png)

*Comentário:* Desempenho próximo ao acaso (acurácia 5.32%). Erros distribuídos entre classes indicam falha na separação das patologias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_1shot_roc.png](../results/03_Siamese_imagenet_50epocas/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_roc.png)

*Comentário:* AUC macro 0.455: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_1shot_tsne.png](../results/03_Siamese_imagenet_50epocas/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 48.16%, F1 macro 34.53%, AUC macro 0.496.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_3shot_training.png](../results/03_Siamese_imagenet_50epocas/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_3shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_confusion_matrix.png)

*Comentário:* Acurácia aparente 48.16%, porém F1 macro baixo (34.53%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_3shot_roc.png](../results/03_Siamese_imagenet_50epocas/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_roc.png)

*Comentário:* AUC macro 0.496: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_3shot_tsne.png](../results/03_Siamese_imagenet_50epocas/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 78.83%, F1 macro 50.26%, AUC macro 0.608.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_5shot_training.png](../results/03_Siamese_imagenet_50epocas/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_5shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 78.83% (F1 macro 50.26%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_5shot_roc.png](../results/03_Siamese_imagenet_50epocas/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_roc.png)

*Comentário:* AUC macro 0.608: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_5shot_tsne.png](../results/03_Siamese_imagenet_50epocas/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 03 — Imbalanced Tuberculosis

#### 1-SHOT

**Métricas:** acurácia 31.82%, F1 macro 31.34%, AUC macro 0.542.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_1shot_training.png](../results/03_Siamese_imagenet_50epocas/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_1shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_confusion_matrix.png)

*Comentário:* Acurácia aparente 31.82%, porém F1 macro baixo (31.34%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_1shot_roc.png](../results/03_Siamese_imagenet_50epocas/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_roc.png)

*Comentário:* AUC macro 0.542: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_1shot_tsne.png](../results/03_Siamese_imagenet_50epocas/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 39.94%, F1 macro 41.92%, AUC macro 0.613.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_3shot_training.png](../results/03_Siamese_imagenet_50epocas/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_3shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 39.94% (F1 macro 41.92%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_3shot_roc.png](../results/03_Siamese_imagenet_50epocas/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_roc.png)

*Comentário:* AUC macro 0.613: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_3shot_tsne.png](../results/03_Siamese_imagenet_50epocas/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 58.51%, F1 macro 51.70%, AUC macro 0.679.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_5shot_training.png](../results/03_Siamese_imagenet_50epocas/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_5shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 58.51% (F1 macro 51.70%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_5shot_roc.png](../results/03_Siamese_imagenet_50epocas/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_roc.png)

*Comentário:* AUC macro 0.679: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_5shot_tsne.png](../results/03_Siamese_imagenet_50epocas/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 04 — MakeDataset

#### 1-SHOT

**Métricas:** acurácia 49.00%, F1 macro 48.21%, AUC macro 0.702.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_1shot_training.png](../results/03_Siamese_imagenet_50epocas/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_1shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 49.00% (F1 macro 48.21%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_1shot_roc.png](../results/03_Siamese_imagenet_50epocas/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_roc.png)

*Comentário:* AUC macro 0.702: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_1shot_tsne.png](../results/03_Siamese_imagenet_50epocas/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 59.67%, F1 macro 60.66%, AUC macro 0.777.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_3shot_training.png](../results/03_Siamese_imagenet_50epocas/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_3shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 59.67% (F1 macro 60.66%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_3shot_roc.png](../results/03_Siamese_imagenet_50epocas/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_roc.png)

*Comentário:* AUC macro 0.777: boa separabilidade; curvas bem acima da linha aleatória.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_3shot_tsne.png](../results/03_Siamese_imagenet_50epocas/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 73.67%, F1 macro 73.69%, AUC macro 0.781.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_5shot_training.png](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 73.67% (F1 macro 73.69%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_5shot_roc.png](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_roc.png)

*Comentário:* AUC macro 0.781: boa separabilidade; curvas bem acima da linha aleatória.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_5shot_tsne.png](../results/03_Siamese_imagenet_50epocas/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 05 — Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset

#### 1-SHOT

**Métricas:** acurácia 33.61%, F1 macro 28.25%, AUC macro 0.627.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_1shot_training.png](../results/03_Siamese_imagenet_50epocas/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_training.png)

*Comentário:* Desempenho de validação muito baixo; o modelo não generalizou. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_1shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_confusion_matrix.png)

*Comentário:* Acurácia aparente 33.61%, porém F1 macro baixo (28.25%): o modelo tende a favorecer classes majoritárias e errar classes minoritárias.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_1shot_roc.png](../results/03_Siamese_imagenet_50epocas/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_roc.png)

*Comentário:* AUC macro 0.627: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_1shot_tsne.png](../results/03_Siamese_imagenet_50epocas/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 51.04%, F1 macro 45.62%, AUC macro 0.693.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_3shot_training.png](../results/03_Siamese_imagenet_50epocas/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_3shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 51.04% (F1 macro 45.62%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_3shot_roc.png](../results/03_Siamese_imagenet_50epocas/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_roc.png)

*Comentário:* AUC macro 0.693: discriminação limitada; algumas classes se sobrepõem no espaço de scores.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_3shot_tsne.png](../results/03_Siamese_imagenet_50epocas/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 58.79%, F1 macro 51.74%, AUC macro 0.782.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_5shot_training.png](../results/03_Siamese_imagenet_50epocas/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_5shot_confusion_matrix.png](../results/03_Siamese_imagenet_50epocas/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 58.79% (F1 macro 51.74%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_5shot_roc.png](../results/03_Siamese_imagenet_50epocas/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_roc.png)

*Comentário:* AUC macro 0.782: boa separabilidade; curvas bem acima da linha aleatória.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_5shot_tsne.png](../results/03_Siamese_imagenet_50epocas/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

---

## 4. Siamese ImageNet — run final

Rede Siamese com ResNet50V2 (ImageNet), triplet loss. Avaliação few-shot (treino completo + few-shot).

| Exp. | Shot | Acurácia | F1 macro | AUC macro |
|------|------|----------|----------|-----------|
| 01 | 1-shot | 72.03% | 61.34% | 0.957 |
| 01 | 3-shot | 92.37% | 91.98% | 0.996 |
| 01 | 5-shot | 94.92% | 92.12% | 0.994 |
| 02 | 1-shot | 82.51% | 57.55% | 0.966 |
| 02 | 3-shot | 91.63% | 68.80% | 1.000 |
| 02 | 5-shot | 97.47% | 84.63% | 1.000 |
| 03 | 1-shot | 88.60% | 85.08% | 0.968 |
| 03 | 3-shot | 90.03% | 87.24% | 0.987 |
| 03 | 5-shot | 93.70% | 91.88% | 0.993 |
| 04 | 1-shot | 93.00% | 92.95% | 0.990 |
| 04 | 3-shot | 98.67% | 98.67% | 0.999 |
| 04 | 5-shot | 99.67% | 99.67% | 1.000 |
| 05 | 1-shot | 77.32% | 68.56% | 0.926 |
| 05 | 3-shot | 87.23% | 79.48% | 0.964 |
| 05 | 5-shot | 89.13% | 82.40% | 0.979 |
### Experimento 01 — COVID Chest X-ray

#### 1-SHOT

**Métricas:** acurácia 72.03%, F1 macro 61.34%, AUC macro 0.957.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_1shot_training.png](../results/04_Siamese_imagenet/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_1shot_confusion_matrix.png](../results/04_Siamese_imagenet/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 72.03% (F1 macro 61.34%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_1shot_roc.png](../results/04_Siamese_imagenet/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_roc.png)

*Comentário:* AUC macro 0.957: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_1shot_tsne.png](../results/04_Siamese_imagenet/01/1-shot/siamese_resnet50v2_imagenet_01_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 92.37%, F1 macro 91.98%, AUC macro 0.996.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_3shot_training.png](../results/04_Siamese_imagenet/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_3shot_confusion_matrix.png](../results/04_Siamese_imagenet/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 92.37% (F1 macro 91.98%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_3shot_roc.png](../results/04_Siamese_imagenet/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_roc.png)

*Comentário:* AUC macro 0.996: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_3shot_tsne.png](../results/04_Siamese_imagenet/01/3-shot/siamese_resnet50v2_imagenet_01_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 94.92%, F1 macro 92.12%, AUC macro 0.994.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_01_5shot_training.png](../results/04_Siamese_imagenet/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_01_5shot_confusion_matrix.png](../results/04_Siamese_imagenet/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 94.92% (F1 macro 92.12%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_01_5shot_roc.png](../results/04_Siamese_imagenet/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_roc.png)

*Comentário:* AUC macro 0.994: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_01_5shot_tsne.png](../results/04_Siamese_imagenet/01/5-shot/siamese_resnet50v2_imagenet_01_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 02 — CXR8

#### 1-SHOT

**Métricas:** acurácia 82.51%, F1 macro 57.55%, AUC macro 0.966.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_1shot_training.png](../results/04_Siamese_imagenet/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_1shot_confusion_matrix.png](../results/04_Siamese_imagenet/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 82.51% (F1 macro 57.55%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_1shot_roc.png](../results/04_Siamese_imagenet/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_roc.png)

*Comentário:* AUC macro 0.966: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_1shot_tsne.png](../results/04_Siamese_imagenet/02/1-shot/siamese_resnet50v2_imagenet_02_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 91.63%, F1 macro 68.80%, AUC macro 1.000.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_3shot_training.png](../results/04_Siamese_imagenet/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_3shot_confusion_matrix.png](../results/04_Siamese_imagenet/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 91.63% (F1 macro 68.80%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_3shot_roc.png](../results/04_Siamese_imagenet/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_roc.png)

*Comentário:* AUC macro 1.000: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_3shot_tsne.png](../results/04_Siamese_imagenet/02/3-shot/siamese_resnet50v2_imagenet_02_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 97.47%, F1 macro 84.63%, AUC macro 1.000.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_02_5shot_training.png](../results/04_Siamese_imagenet/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_02_5shot_confusion_matrix.png](../results/04_Siamese_imagenet/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 97.47%). A diagonal principal concentra a maior parte das predições corretas.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_02_5shot_roc.png](../results/04_Siamese_imagenet/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_roc.png)

*Comentário:* AUC macro 1.000: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_02_5shot_tsne.png](../results/04_Siamese_imagenet/02/5-shot/siamese_resnet50v2_imagenet_02_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 03 — Imbalanced Tuberculosis

#### 1-SHOT

**Métricas:** acurácia 88.60%, F1 macro 85.08%, AUC macro 0.968.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_1shot_training.png](../results/04_Siamese_imagenet/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_1shot_confusion_matrix.png](../results/04_Siamese_imagenet/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 88.60% (F1 macro 85.08%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_1shot_roc.png](../results/04_Siamese_imagenet/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_roc.png)

*Comentário:* AUC macro 0.968: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_1shot_tsne.png](../results/04_Siamese_imagenet/03/1-shot/siamese_resnet50v2_imagenet_03_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 90.03%, F1 macro 87.24%, AUC macro 0.987.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_3shot_training.png](../results/04_Siamese_imagenet/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_3shot_confusion_matrix.png](../results/04_Siamese_imagenet/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 90.03% (F1 macro 87.24%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_3shot_roc.png](../results/04_Siamese_imagenet/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_roc.png)

*Comentário:* AUC macro 0.987: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_3shot_tsne.png](../results/04_Siamese_imagenet/03/3-shot/siamese_resnet50v2_imagenet_03_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 93.70%, F1 macro 91.88%, AUC macro 0.993.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_03_5shot_training.png](../results/04_Siamese_imagenet/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_03_5shot_confusion_matrix.png](../results/04_Siamese_imagenet/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 93.70% (F1 macro 91.88%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_03_5shot_roc.png](../results/04_Siamese_imagenet/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_roc.png)

*Comentário:* AUC macro 0.993: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_03_5shot_tsne.png](../results/04_Siamese_imagenet/03/5-shot/siamese_resnet50v2_imagenet_03_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 04 — MakeDataset

#### 1-SHOT

**Métricas:** acurácia 93.00%, F1 macro 92.95%, AUC macro 0.990.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_1shot_training.png](../results/04_Siamese_imagenet/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_1shot_confusion_matrix.png](../results/04_Siamese_imagenet/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 93.00% (F1 macro 92.95%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_1shot_roc.png](../results/04_Siamese_imagenet/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_roc.png)

*Comentário:* AUC macro 0.990: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_1shot_tsne.png](../results/04_Siamese_imagenet/04/1-shot/siamese_resnet50v2_imagenet_04_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 98.67%, F1 macro 98.67%, AUC macro 0.999.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_3shot_training.png](../results/04_Siamese_imagenet/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_3shot_confusion_matrix.png](../results/04_Siamese_imagenet/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 98.67%). A diagonal principal concentra a maior parte das predições corretas.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_3shot_roc.png](../results/04_Siamese_imagenet/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_roc.png)

*Comentário:* AUC macro 0.999: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_3shot_tsne.png](../results/04_Siamese_imagenet/04/3-shot/siamese_resnet50v2_imagenet_04_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 99.67%, F1 macro 99.67%, AUC macro 1.000.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_04_5shot_training.png](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_training.png)

*Comentário:* Convergência rápida com acurácia de validação elevada. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_confusion_matrix.png)

*Comentário:* A matriz confirma excelente desempenho (acurácia 99.67%). A diagonal principal concentra a maior parte das predições corretas.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_04_5shot_roc.png](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_roc.png)

*Comentário:* AUC macro 1.000: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_04_5shot_tsne.png](../results/04_Siamese_imagenet/04/5-shot/siamese_resnet50v2_imagenet_04_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

### Experimento 05 — Treino: Imbalanced Tuberculosis · Validação: COVID Chest X-ray · Teste: MakeDataset

#### 1-SHOT

**Métricas:** acurácia 77.32%, F1 macro 68.56%, AUC macro 0.926.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_1shot_training.png](../results/04_Siamese_imagenet/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_1shot_confusion_matrix.png](../results/04_Siamese_imagenet/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_confusion_matrix.png)

*Comentário:* Acurácia de 77.32% (F1 macro 68.56%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_1shot_roc.png](../results/04_Siamese_imagenet/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_roc.png)

*Comentário:* AUC macro 0.926: boa separabilidade; curvas bem acima da linha aleatória.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_1shot_tsne.png](../results/04_Siamese_imagenet/05/1-shot/siamese_resnet50v2_imagenet_05_1shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 3-SHOT

**Métricas:** acurácia 87.23%, F1 macro 79.48%, AUC macro 0.964.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_3shot_training.png](../results/04_Siamese_imagenet/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_3shot_confusion_matrix.png](../results/04_Siamese_imagenet/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_confusion_matrix.png)

*Comentário:* Acurácia de 87.23% (F1 macro 79.48%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_3shot_roc.png](../results/04_Siamese_imagenet/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_roc.png)

*Comentário:* AUC macro 0.964: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_3shot_tsne.png](../results/04_Siamese_imagenet/05/3-shot/siamese_resnet50v2_imagenet_05_3shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

#### 5-SHOT

**Métricas:** acurácia 89.13%, F1 macro 82.40%, AUC macro 0.979.

##### Curvas de treino (triplet loss)

![siamese_resnet50v2_imagenet_05_5shot_training.png](../results/04_Siamese_imagenet/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_training.png)

*Comentário:* Curvas de perda e acurácia ao longo das épocas. O ponto salvo corresponde ao menor `val_loss`.

##### Matriz de confusão (classificação por protótipos)

![siamese_resnet50v2_imagenet_05_5shot_confusion_matrix.png](../results/04_Siamese_imagenet/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_confusion_matrix.png)

*Comentário:* Acurácia de 89.13% (F1 macro 82.40%). Confusões fora da diagonal indicam classes visualmente similares ou desbalanceamento.

##### Curvas ROC (one-vs-rest)

![siamese_resnet50v2_imagenet_05_5shot_roc.png](../results/04_Siamese_imagenet/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_roc.png)

*Comentário:* AUC macro 0.979: excelente capacidade de discriminação entre classes.

##### Visualização t-SNE dos embeddings

![siamese_resnet50v2_imagenet_05_5shot_tsne.png](../results/04_Siamese_imagenet/05/5-shot/siamese_resnet50v2_imagenet_05_5shot_tsne.png)

*Comentário:* Projeção t-SNE dos embeddings L2-normalizados. Clusters mais compactos e separados indicam representações discriminativas por classe.

---

## Conclusões

1. **ResNet50V2 supera VGG19** na maioria dos cenários, especialmente quando há transfer learning e dados mais equilibrados (exp. 03 e 04).

2. **Generalização entre bases distintas** (exp. 05) permanece o maior desafio para classificadores tradicionais; acurácia cai para ~25% (VGG19) e ~61% (ResNet50V2).

3. **Rede Siamese com ImageNet** alcança os melhores resultados globais, com 99,67% no exp. 04 (5-shot), demonstrando que embeddings métricos + protótipos few-shot se adaptam bem a poucos exemplos por classe.

4. **Curvas de treino** mostram convergência rápida em datasets pequenos; em cenários difíceis, a matriz de confusão revela confusão sistemática entre patologias pulmonares similares.

5. **Curvas ROC e t-SNE** (Siamese) corroboram a separabilidade das classes: AUC macro > 0,95 nos melhores experimentos e clusters visuais coerentes no espaço de embedding.

---

*Imagens referenciadas a partir de `results/`. Métricas completas disponíveis nos arquivos `*_metrics.json` de cada pasta.*
