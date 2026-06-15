import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))

from config import CLASS_NAMES, EXTENSOES, SIAMESE_PROTO_VIEWS, SPLIT_DIR
from image_loader import carregar_imagem_rgb
from plots import evaluate_and_plot, plot_training
from siamese_transforms import augmentar_prototipo_tf

PAIR_DIR = ROOT / "datasets" / "pair"

# Carrega a imagem.
def _ler_imagem(caminho: str) -> np.ndarray:
    return carregar_imagem_rgb(caminho)

# Lista as imagens.
def _listar_imagens(pasta: Path) -> list[str]:
    if not pasta.is_dir():
        return []
    return [
        str(arquivo.resolve())
        for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
    ]

# Coleta as imagens de teste.
def _coletar_teste(dataset: str) -> tuple[list[str], list[int]]:
    """Coleta imagens de teste do split com labels conhecidos."""
    arquivos = []
    labels = []
    pasta_test = SPLIT_DIR / dataset / "test"

    if not pasta_test.is_dir():
        return arquivos, labels

    for indice, classe in enumerate(CLASS_NAMES):
        pasta_classe = pasta_test / classe
        for caminho in _listar_imagens(pasta_classe):
            arquivos.append(caminho)
            labels.append(indice)

    return arquivos, labels

# Normaliza o vetor.
def _normalizar_vetor(vetor: np.ndarray) -> np.ndarray:
    norma = float(np.linalg.norm(vetor))
    if norma < 1e-8:
        return vetor
    return vetor / norma

# Extrai o embedding da imagem.
def _embedding_tensor(rede, imagem: tf.Tensor) -> np.ndarray:
    tensor = tf.expand_dims(imagem, axis=0)
    return _normalizar_vetor(rede.predict(tensor, verbose=0)[0])

# Extrai o embedding da ancora.
def _embedding_ancora(rede, caminho: str, n_views: int = SIAMESE_PROTO_VIEWS) -> np.ndarray:
    imagem_base = tf.convert_to_tensor(_ler_imagem(caminho), dtype=tf.float32)
    embeddings = []

    for vista in range(max(1, n_views)):
        if vista == 0:
            imagem = imagem_base
        else:
            imagem = augmentar_prototipo_tf(imagem_base, semente=hash(caminho) % 10_000 + vista)
        embeddings.append(_embedding_tensor(rede, imagem))

    return _normalizar_vetor(np.mean(embeddings, axis=0))

# Cria o banco de suporte.
def _banco_suporte(rede, numero_pair: str, shot: int) -> dict[str, list[np.ndarray]]:
    # Todas as ancoras (com vistas) — base para k-NN few-shot.
    pasta_shot = PAIR_DIR / numero_pair / "ancora" / f"{shot}-shot"
    banco: dict[str, list[np.ndarray]] = {}

    for classe in CLASS_NAMES:
        caminhos = _listar_imagens(pasta_shot / classe)
        if not caminhos:
            continue
        banco[classe] = [_embedding_ancora(rede, caminho) for caminho in caminhos]

    return banco

# Cria os prototipos de banco.
def _prototipos_de_banco(banco: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    prototipos = {}
    for classe, vetores in banco.items():
        prototipos[classe] = _normalizar_vetor(np.mean(vetores, axis=0))
    return prototipos

# Cria os prototipos de banco por classe.
def prototipos_por_classe(rede, numero_pair: str, shot: int) -> dict[str, np.ndarray]:
    """Prototipos few-shot por classe (ancoras + vistas augmentadas)."""
    banco = _banco_suporte(rede, numero_pair, shot)
    return _prototipos_de_banco(banco)

# Extrai o embedding da imagem.
def _embedding(rede, caminho_imagem: str) -> np.ndarray:
    imagem = np.expand_dims(_ler_imagem(caminho_imagem), axis=0)
    return _normalizar_vetor(rede.predict(imagem, verbose=0)[0])

# Calcula a distância mínima entre o embedding e os vetores da classe.
def _distancia_minima_classe(embedding: np.ndarray, vetores: list[np.ndarray]) -> float:
    return float(min(np.linalg.norm(embedding - vetor) for vetor in vetores))

# Calcula os scores e a predição.
def _scores_e_predicao(
    embedding: np.ndarray,
    prototipos: dict[str, np.ndarray],
    banco: dict[str, list[np.ndarray]],
) -> tuple[int, np.ndarray]:
    
    # Classificacao hibrida: menor distancia entre prototipo e k-NN (ancoras).
    scores = np.full(len(CLASS_NAMES), -np.inf, dtype=np.float64)
    melhor_indice = 0
    menor_distancia = float("inf")

    for classe in CLASS_NAMES:
        if classe not in banco:
            continue

        indice = CLASS_NAMES.index(classe)
        dist_knn = _distancia_minima_classe(embedding, banco[classe])
        dist_proto = float(np.linalg.norm(embedding - prototipos[classe]))
        distancia = min(dist_knn, dist_proto)
        scores[indice] = -distancia

        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_indice = indice

    return melhor_indice, scores

# Avalia a classificação few-shot.
def avaliar_classificacao(
    rede,
    numero_pair: str,
    nome_dataset: str,
    shot: int,
    prefixo: str,
    pasta_saida: Path,
):
    
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    banco = _banco_suporte(rede, numero_pair, shot)
    prototipos = _prototipos_de_banco(banco)
    if not prototipos:
        print("[aviso] Nenhuma ancora encontrada para avaliacao few-shot.")
        return None

    arquivos, y_true = _coletar_teste(nome_dataset)
    if not arquivos:
        print(f"[aviso] Teste nao encontrado: {SPLIT_DIR / nome_dataset / 'test'}")
        return None

    print(f"\nAvaliando classificacao few-shot ({shot}-shot)...")
    print(f"Suporte: {list(banco.keys())} | classificador: prototipo + k-NN")
    print(f"Imagens de teste: {len(arquivos)}")

    y_pred = []
    y_scores = []

    for caminho in arquivos:
        embedding = _embedding(rede, caminho)
        predicao, scores = _scores_e_predicao(embedding, prototipos, banco)
        y_pred.append(predicao)
        y_scores.append(scores)

    metricas = evaluate_and_plot(
        np.array(y_true),
        np.array(y_pred),
        CLASS_NAMES,
        prefixo,
        pasta_saida,
        y_score=np.array(y_scores),
    )
    return metricas

# Salva o gráfico de treino.
def salvar_grafico_treino(historico, prefixo: str, pasta_saida: Path):
    plot_training(historico, prefixo, pasta_saida)
