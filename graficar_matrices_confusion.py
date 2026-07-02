import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grafica matrices de confusion de los tests DeFaX."
    )
    parser.add_argument(
        "--original",
        default=r"checkpoints\metricas_test_original.json",
        help="JSON del test original del entrenamiento.",
    )
    parser.add_argument(
        "--externo",
        default=r"checkpoints\metricas_test_externo.json",
        help="JSON del test externo.",
    )
    parser.add_argument(
        "--salida",
        default=r"checkpoints\matrices_confusion.png",
        help="Ruta donde se guardara la imagen.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def cargar_metricas(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        metricas = json.load(f)

    matriz = np.asarray(metricas["confusion_matrix"], dtype=int)
    labels = metricas.get("confusion_matrix_labels", ["real", "fake"])
    return path, metricas, matriz, labels


def anotar_celda(ax, fila, columna, valor, maximo):
    color = "white" if valor > maximo * 0.55 else "black"
    ax.text(
        columna,
        fila,
        f"{valor:,}",
        ha="center",
        va="center",
        color=color,
        fontsize=10,
        fontweight="bold",
    )


def dibujar_matriz(ax, titulo, metricas, matriz, labels):
    imagen = ax.imshow(matriz, cmap="Blues")
    maximo = int(matriz.max())

    ax.set_title(
        f"{titulo}\nacc={metricas['accuracy']:.4f}  f1={metricas['f1']:.4f}  auc={metricas['auc']:.4f}",
        fontsize=11,
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)

    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            anotar_celda(ax, fila, columna, int(matriz[fila, columna]), maximo)

    ax.text(
        0.5,
        -0.28,
        f"TN={metricas['tn']:,}  FP={metricas['fp']:,}  FN={metricas['fn']:,}  TP={metricas['tp']:,}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    return imagen


def main():
    args = parse_args()
    original_path, original, matriz_original, labels_original = cargar_metricas(args.original)
    externo_path, externo, matriz_externo, labels_externo = cargar_metricas(args.externo)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    img_original = dibujar_matriz(
        axes[0],
        f"Test original\n{original_path.name}",
        original,
        matriz_original,
        labels_original,
    )
    dibujar_matriz(
        axes[1],
        f"Test externo\n{externo_path.name}",
        externo,
        matriz_externo,
        labels_externo,
    )

    fig.colorbar(img_original, ax=axes, shrink=0.82, label="Cantidad")
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=args.dpi)
    plt.close(fig)
    print(f"grafico guardado: {salida}")


if __name__ == "__main__":
    main()
