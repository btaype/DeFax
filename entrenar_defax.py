import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datos_rgb_video import construir_dataloaders, imprimir_resumen_splits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento DeFaX RGB con split 72/14/14 por video."
    )
    parser.add_argument(
        "--data-root",
        default=r"F:\PROYECTOf1PAPER4\Paper4geometria\data_completa",
    )
    parser.add_argument("--salida", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--swin-modelo", default="swin_base_patch4_window7_224")
    parser.add_argument("--efficient-modelo", default="efficientnet_b0")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--sin-balanceo", action="store_true")
    parser.add_argument("--imprimir-shapes", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--guardar-cada-epoca", action="store_true")
    return parser.parse_args()


def fijar_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def mover_batch(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def calcular_metricas(y_true, probs, loss):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    p_fake = probs[:, 1]

    metricas = {
        "loss": loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }

    try:
        metricas["auc"] = roc_auc_score(y_true, p_fake)
    except ValueError:
        metricas["auc"] = float("nan")

    try:
        metricas["log_loss"] = log_loss(y_true, probs, labels=[0, 1])
    except ValueError:
        metricas["log_loss"] = float("nan")

    return metricas


def entrenar_una_epoca(modelo, loader, criterio, optimizador, device, epoch, total_epochs):
    modelo.train()
    perdida_total = 0.0
    correctos = 0
    vistos = 0
    y_true = []
    probs = []

    barra = tqdm(loader, desc=f"train {epoch}/{total_epochs}", leave=True)
    for batch in barra:
        x, y = mover_batch(batch, device)

        optimizador.zero_grad(set_to_none=True)
        logits = modelo(x)
        loss = criterio(logits, y)
        loss.backward()
        optimizador.step()

        perdida_total += loss.item() * x.size(0)
        prob = torch.softmax(logits.detach(), dim=1)
        pred = prob.argmax(dim=1)
        correctos += (pred == y).sum().item()
        vistos += y.numel()
        y_true.extend(y.detach().cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())

        barra.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correctos / max(vistos, 1):.4f}",
        )

    loss_promedio = perdida_total / len(loader.dataset)
    return calcular_metricas(y_true, probs, loss_promedio)


@torch.no_grad()
def evaluar(modelo, loader, criterio, device, nombre):
    modelo.eval()
    perdida_total = 0.0
    y_true = []
    probs = []

    barra = tqdm(loader, desc=nombre, leave=True)
    for batch in barra:
        x, y = mover_batch(batch, device)
        logits = modelo(x)
        loss = criterio(logits, y)

        perdida_total += loss.item() * x.size(0)
        prob = torch.softmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        probs.extend(prob.cpu().numpy().tolist())

    loss_promedio = perdida_total / len(loader.dataset)
    return calcular_metricas(y_true, probs, loss_promedio)


def imprimir_metricas(nombre, metricas):
    partes = [
        f"{nombre}",
        f"loss={metricas['loss']:.4f}",
        f"acc={metricas['accuracy']:.4f}",
        f"precision={metricas['precision']:.4f}",
        f"recall={metricas['recall']:.4f}",
        f"f1={metricas['f1']:.4f}",
        f"auc={metricas['auc']:.4f}",
        f"log_loss={metricas['log_loss']:.4f}",
        f"kappa={metricas['kappa']:.4f}",
    ]
    print(" | ".join(partes))


def guardar_checkpoint(
    path,
    modelo,
    optimizador,
    scheduler,
    epoch,
    train_metricas,
    val_metricas,
    mejor_f1,
    sin_mejora,
    args,
):
    payload = {
        "epoch": epoch,
        "model_state": modelo.state_dict(),
        "optimizer_state": optimizador.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_metricas": train_metricas,
        "val_metricas": val_metricas,
        "mejor_f1": mejor_f1,
        "sin_mejora": sin_mejora,
        "args": vars(args),
        "label_map": {"real": 0, "fake": 1},
    }
    torch.save(payload, path)


def cargar_checkpoint(path, modelo, optimizador, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    modelo.load_state_dict(checkpoint["model_state"])
    optimizador.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    inicio_epoch = int(checkpoint["epoch"]) + 1
    mejor_f1 = float(checkpoint.get("mejor_f1", checkpoint["val_metricas"]["f1"]))
    sin_mejora = int(checkpoint.get("sin_mejora", 0))
    return inicio_epoch, mejor_f1, sin_mejora


def fila_metricas(epoch, lr_actual, train_metricas, val_metricas):
    fila = {
        "epoch": epoch,
        "lr": lr_actual,
    }
    for prefijo, metricas in (("train", train_metricas), ("val", val_metricas)):
        for clave in (
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "log_loss",
            "kappa",
        ):
            fila[f"{prefijo}_{clave}"] = metricas[clave]
    return fila


def guardar_metricas(path_csv, path_jsonl, fila):
    campos = list(fila.keys())
    existe = path_csv.exists()

    with path_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        if not existe:
            writer.writeheader()
        writer.writerow(fila)

    with path_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=True) + "\n")


def guardar_test_final(path, metricas):
    with path.open("w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=True)


def main():
    args = parse_args()
    fijar_seed(args.seed)

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    datasets, loaders, splits_video = construir_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        balancear_train=not args.sin_balanceo,
    )
    imprimir_resumen_splits(datasets, splits_video)

    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.cuda.get_device_name(0)}")

    try:
        from modelos.defax import DeFaX
    except ModuleNotFoundError as exc:
        if exc.name == "timm":
            raise ModuleNotFoundError(
                "Falta instalar timm. Instala las dependencias con: "
                "pip install -r requirements.txt"
            ) from exc
        raise

    modelo = DeFaX(
        swin_modelo=args.swin_modelo,
        efficient_modelo=args.efficient_modelo,
        pretrained=not args.no_pretrained,
        imprimir_shapes=args.imprimir_shapes,
    ).to(device)

    criterio = nn.CrossEntropyLoss()
    optimizador = AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizador,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )

    mejor_f1 = -1.0
    sin_mejora = 0
    inicio_epoch = 1
    best_path = salida / "best_defax.pth"
    last_path = salida / "last_defax.pth"
    metricas_csv = salida / "metricas_por_epoca.csv"
    metricas_jsonl = salida / "metricas_por_epoca.jsonl"

    if args.resume is not None:
        inicio_epoch, mejor_f1, sin_mejora = cargar_checkpoint(
            args.resume,
            modelo,
            optimizador,
            scheduler,
            device,
        )
        print(f"continuando desde {args.resume}, siguiente epoch: {inicio_epoch}")

    for epoch in range(inicio_epoch, args.epochs + 1):
        train_metricas = entrenar_una_epoca(
            modelo,
            loaders["train"],
            criterio,
            optimizador,
            device,
            epoch,
            args.epochs,
        )
        val_metricas = evaluar(modelo, loaders["val"], criterio, device, "val")
        lr_actual = optimizador.param_groups[0]["lr"]
        scheduler.step()

        imprimir_metricas(f"epoch {epoch} train", train_metricas)
        imprimir_metricas(f"epoch {epoch} val", val_metricas)

        fila = fila_metricas(epoch, lr_actual, train_metricas, val_metricas)
        guardar_metricas(metricas_csv, metricas_jsonl, fila)
        print(f"metricas guardadas: {metricas_csv}")

        if val_metricas["f1"] > mejor_f1:
            mejor_f1 = val_metricas["f1"]
            sin_mejora = 0
            guardar_checkpoint(
                best_path,
                modelo,
                optimizador,
                scheduler,
                epoch,
                train_metricas,
                val_metricas,
                mejor_f1,
                sin_mejora,
                args,
            )
            print(f"checkpoint guardado: {best_path}")
        else:
            sin_mejora += 1
            print(f"sin mejora: {sin_mejora}/{args.patience}")

        guardar_checkpoint(
            last_path,
            modelo,
            optimizador,
            scheduler,
            epoch,
            train_metricas,
            val_metricas,
            mejor_f1,
            sin_mejora,
            args,
        )
        print(f"ultimo checkpoint guardado: {last_path}")

        if args.guardar_cada_epoca:
            epoch_path = salida / f"epoch_{epoch:03d}.pth"
            guardar_checkpoint(
                epoch_path,
                modelo,
                optimizador,
                scheduler,
                epoch,
                train_metricas,
                val_metricas,
                mejor_f1,
                sin_mejora,
                args,
            )
            print(f"checkpoint de epoca guardado: {epoch_path}")

        if sin_mejora >= args.patience:
            print("early stopping activado")
            break

    checkpoint = torch.load(best_path, map_location=device)
    modelo.load_state_dict(checkpoint["model_state"])
    test_metricas = evaluar(modelo, loaders["test"], criterio, device, "test")
    imprimir_metricas("test final", test_metricas)
    guardar_test_final(salida / "metricas_test_final.json", test_metricas)
    print(f"metricas test guardadas: {salida / 'metricas_test_final.json'}")


if __name__ == "__main__":
    main()
