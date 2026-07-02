import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

from datos_rgb_video import construir_dataloaders, imprimir_resumen_splits
from entrenar_defax import calcular_metricas, imprimir_metricas
from modelos.defax import DeFaX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test de DeFaX sobre el split test original del entrenamiento."
    )
    parser.add_argument(
        "--data-root",
        default=r"F:\PROYECTOf1PAPER4\Paper4geometria\data_completa",
    )
    parser.add_argument("--checkpoint", default=r"checkpoints\last_defax.pth")
    parser.add_argument("--salida", default=r"checkpoints\metricas_test_original.json")
    parser.add_argument(
        "--salida-csv",
        default=None,
        help="CSV con metricas por grupo. Por defecto usa el mismo nombre de --salida con extension .csv.",
    )
    parser.add_argument(
        "--salida-dir",
        default=None,
        help="Carpeta donde guardar metricas_test_original.json y metricas_test_original.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swin-modelo", default="swin_base_patch4_window7_224")
    parser.add_argument("--efficient-modelo", default="efficientnet_b0")
    return parser.parse_args()


def cargar_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def mover_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def agregar_matriz_confusion(metricas, y_true, probs):
    y_pred = [max(range(len(prob)), key=prob.__getitem__) for prob in probs]
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    metricas.update(
        {
            "confusion_matrix_labels": ["real", "fake"],
            "confusion_matrix": matriz.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    )
    return metricas


def calcular_metricas_grupo(nombre, y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()

    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "grupo": nombre,
        "n_muestras": int(len(y_true)),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def construir_metricas_por_grupo(dataset, y_true, probs):
    grupos = {"general": list(range(len(y_true)))}

    for idx, sample in enumerate(dataset.samples):
        tipo = sample["tipo"]
        if tipo == "real":
            grupos.setdefault("real", []).append(idx)
        else:
            grupos.setdefault("fake", []).append(idx)
            grupos.setdefault(f"fake_{tipo}", []).append(idx)

    filas = []
    for nombre in ["general", "real", "fake"]:
        indices = grupos.get(nombre, [])
        if indices:
            filas.append(
                calcular_metricas_grupo(
                    nombre,
                    [y_true[i] for i in indices],
                    [probs[i] for i in indices],
                )
            )

    tecnicas = sorted(nombre for nombre in grupos if nombre.startswith("fake_"))
    filas_tecnicas = []
    for nombre in tecnicas:
        indices = grupos[nombre]
        fila = calcular_metricas_grupo(
            nombre,
            [y_true[i] for i in indices],
            [probs[i] for i in indices],
        )
        filas_tecnicas.append(fila)
        filas.append(fila)

    if filas_tecnicas:
        fake_total = next(fila for fila in filas if fila["grupo"] == "fake")
        filas.append(
            {
                "grupo": "promedio_fake_tecnicas",
                "n_muestras": fake_total["n_muestras"],
                "acc": sum(fila["acc"] for fila in filas_tecnicas) / len(filas_tecnicas),
                "f1": sum(fila["f1"] for fila in filas_tecnicas) / len(filas_tecnicas),
                "auc": sum(fila["auc"] for fila in filas_tecnicas) / len(filas_tecnicas),
                "tn": fake_total["tn"],
                "fp": fake_total["fp"],
                "fn": fake_total["fn"],
                "tp": fake_total["tp"],
            }
        )

    return filas


def imprimir_matriz_confusion(metricas):
    matriz = metricas["confusion_matrix"]
    print("matriz confusion filas=real/fake columnas=real/fake")
    print(f"  real: {matriz[0]}")
    print(f"  fake: {matriz[1]}")


@torch.no_grad()
def evaluar(modelo, loader, criterio, device):
    modelo.eval()
    perdida_total = 0.0
    y_true = []
    probs = []

    barra = tqdm(loader, desc="test original", leave=True)
    for batch in barra:
        x, y = mover_batch(batch, device)
        logits = modelo(x)
        loss = criterio(logits, y)

        perdida_total += loss.item() * x.size(0)
        prob = torch.softmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())

    loss_promedio = perdida_total / len(loader.dataset)
    metricas = calcular_metricas(y_true, probs, loss_promedio)
    return agregar_matriz_confusion(metricas, y_true, probs), y_true, probs


def guardar_json(path, metricas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=True)


def guardar_csv(path, filas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    campos = ["grupo", "n_muestras", "acc", "f1", "auc", "tn", "fp", "fn", "tp"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)


def resolver_salidas(args):
    if args.salida_dir is not None:
        salida_dir = Path(args.salida_dir)
        return (
            salida_dir / "metricas_test_original.json",
            salida_dir / "metricas_test_original.csv",
        )

    salida_json = Path(args.salida)
    if salida_json.suffix.lower() != ".json":
        salida_dir = salida_json
        return (
            salida_dir / "metricas_test_original.json",
            salida_dir / "metricas_test_original.csv",
        )

    salida_csv = Path(args.salida_csv) if args.salida_csv else salida_json.with_suffix(".csv")
    return salida_json, salida_csv


def main():
    args = parse_args()
    device = torch.device(args.device)

    datasets, loaders, splits_video = construir_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balancear_train=False,
    )
    imprimir_resumen_splits(datasets, splits_video)

    print(f"data root: {Path(args.data_root)}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.cuda.get_device_name(0)}")

    modelo = DeFaX(
        swin_modelo=args.swin_modelo,
        efficient_modelo=args.efficient_modelo,
        pretrained=False,
    ).to(device)

    checkpoint = cargar_checkpoint(args.checkpoint, device)
    state_dict = checkpoint.get("model_state", checkpoint)
    modelo.load_state_dict(state_dict)

    criterio = nn.CrossEntropyLoss()
    metricas, y_true, probs = evaluar(modelo, loaders["test"], criterio, device)
    filas_csv = construir_metricas_por_grupo(datasets["test"], y_true, probs)
    salida_json, salida_csv = resolver_salidas(args)

    imprimir_metricas("test original", metricas)
    imprimir_matriz_confusion(metricas)
    guardar_json(salida_json, metricas)
    guardar_csv(salida_csv, filas_csv)
    print(f"metricas guardadas: {salida_json}")
    print(f"metricas por grupo guardadas: {salida_csv}")


if __name__ == "__main__":
    main()
