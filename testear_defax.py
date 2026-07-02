import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datos_rgb_video import IMG_EXTS, build_transform
from entrenar_defax import calcular_metricas, imprimir_metricas
from modelos.defax import DeFaX


class RealFakeFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for nombre, etiqueta in (("real", 0), ("fake", 1)):
            clase_dir = self.root_dir / nombre
            if not clase_dir.is_dir():
                raise FileNotFoundError(f"No existe carpeta {nombre}: {clase_dir}")

            imagenes = sorted(
                p
                for p in clase_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )
            for img_path in imagenes:
                self.samples.append((img_path, etiqueta))

        if not self.samples:
            raise RuntimeError(f"No se encontraron imagenes en {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, etiqueta = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(etiqueta, dtype=torch.long)

    def conteo_clases(self):
        conteo = {"real": 0, "fake": 0}
        for _, etiqueta in self.samples:
            conteo["fake" if etiqueta == 1 else "real"] += 1
        return conteo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test externo de DeFaX sobre carpetas real/fake."
    )
    parser.add_argument(
        "--data-root",
        default=r"F:\PROYECTOf1PAPER4\Paper4geometria\data_completa2\test",
        help="Carpeta que contiene real/ y fake/.",
    )
    parser.add_argument("--checkpoint", default=r"checkpoints\last_defax.pth")
    parser.add_argument("--salida", default=r"checkpoints\metricas_test_externo.json")
    parser.add_argument(
        "--salida-csv",
        default=None,
        help="CSV con metricas generales y matriz de confusion. Por defecto usa el mismo nombre de --salida con extension .csv.",
    )
    parser.add_argument(
        "--salida-dir",
        default=None,
        help="Carpeta donde guardar metricas_test_externo.json y metricas_test_externo.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
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

    barra = tqdm(loader, desc="test externo", leave=True)
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
    return agregar_matriz_confusion(metricas, y_true, probs)


def guardar_json(path, metricas):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=True)


def guardar_csv(path, metricas, n_muestras):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    campos = ["grupo", "n_muestras", "acc", "f1", "auc", "tn", "fp", "fn", "tp"]
    fila = {
        "grupo": "general",
        "n_muestras": n_muestras,
        "acc": metricas["accuracy"],
        "f1": metricas["f1"],
        "auc": metricas["auc"],
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
        "tp": metricas["tp"],
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerow(fila)


def resolver_salidas(args):
    if args.salida_dir is not None:
        salida_dir = Path(args.salida_dir)
        return (
            salida_dir / "metricas_test_externo.json",
            salida_dir / "metricas_test_externo.csv",
        )

    salida_json = Path(args.salida)
    if salida_json.suffix.lower() != ".json":
        salida_dir = salida_json
        return (
            salida_dir / "metricas_test_externo.json",
            salida_dir / "metricas_test_externo.csv",
        )

    salida_csv = Path(args.salida_csv) if args.salida_csv else salida_json.with_suffix(".csv")
    return salida_json, salida_csv


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = RealFakeFolderDataset(
        args.data_root,
        transform=build_transform("test"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"data root: {Path(args.data_root)}")
    print(f"imagenes: {len(dataset)}")
    print(f"clases: {dataset.conteo_clases()}")
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
    metricas = evaluar(modelo, loader, criterio, device)
    salida_json, salida_csv = resolver_salidas(args)

    imprimir_metricas("test externo", metricas)
    imprimir_matriz_confusion(metricas)
    guardar_json(salida_json, metricas)
    guardar_csv(salida_csv, metricas, len(dataset))
    print(f"metricas guardadas: {salida_json}")
    print(f"metricas csv guardadas: {salida_csv}")


if __name__ == "__main__":
    main()
