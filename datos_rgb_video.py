import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoItem:
    ruta: Path
    etiqueta: int
    tipo: str
    video_id: str


def _imagenes_de_video(video_dir):
    return sorted(
        (
            p
            for p in video_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ),
        key=_clave_frame,
    )


def _clave_video_id(video_id):
    partes = re.split(r"(\d+)", video_id)
    return tuple(int(parte) if parte.isdigit() else parte.lower() for parte in partes)


def _clave_frame(path):
    partes = re.split(r"(\d+)", path.stem)
    return tuple(int(parte) if parte.isdigit() else parte.lower() for parte in partes)


def _seleccionar_uniforme(imagenes, porcentaje_datos):
    total = len(imagenes)
    if total == 0:
        return []

    cantidad = max(1, round(total * porcentaje_datos / 100))
    cantidad = min(cantidad, total)
    if cantidad == total:
        return imagenes
    if cantidad == 1:
        return [imagenes[0]]

    indices = [
        round(i * (total - 1) / (cantidad - 1))
        for i in range(cantidad)
    ]
    return [imagenes[indice] for indice in indices]


def _dividir_video_ids(video_ids, proporciones=(0.72, 0.14, 0.14)):
    video_ids = sorted(set(video_ids), key=_clave_video_id)

    n = len(video_ids)
    n_train = round(n * proporciones[0])
    n_val = round(n * proporciones[1])
    n_test = n - n_train - n_val

    if n_test < 0:
        n_val += n_test
        n_test = 0

    return {
        "train": set(video_ids[:n_train]),
        "val": set(video_ids[n_train : n_train + n_val]),
        "test": set(video_ids[n_train + n_val : n_train + n_val + n_test]),
    }


def construir_splits_por_video(root_dir, seed=42):
    root = Path(root_dir)
    real_root = root / "real"
    fake_root = root / "fake"

    if not real_root.is_dir():
        raise FileNotFoundError(f"No existe carpeta real: {real_root}")
    if not fake_root.is_dir():
        raise FileNotFoundError(f"No existe carpeta fake: {fake_root}")

    videos_real = sorted(p for p in real_root.iterdir() if p.is_dir())
    tecnicas_fake = sorted(p for p in fake_root.iterdir() if p.is_dir())

    todos_video_ids = {video_dir.name for video_dir in videos_real}
    videos_fake_por_tecnica = {}
    for tecnica_dir in tecnicas_fake:
        videos_fake = sorted(p for p in tecnica_dir.iterdir() if p.is_dir())
        videos_fake_por_tecnica[tecnica_dir] = videos_fake
        todos_video_ids.update(video_dir.name for video_dir in videos_fake)

    ids_por_split = _dividir_video_ids(todos_video_ids)
    splits = {"train": [], "val": [], "test": []}

    for split, video_ids in ids_por_split.items():
        for video_dir in videos_real:
            if video_dir.name not in video_ids:
                continue
            splits[split].append(
                VideoItem(
                    ruta=video_dir,
                    etiqueta=0,
                    tipo="real",
                    video_id=video_dir.name,
                )
            )

        for tecnica_dir, videos_fake in videos_fake_por_tecnica.items():
            for video_dir in videos_fake:
                if video_dir.name not in video_ids:
                    continue
                splits[split].append(
                    VideoItem(
                        ruta=video_dir,
                        etiqueta=1,
                        tipo=tecnica_dir.name,
                        video_id=video_dir.name,
                    )
                )

    assert ids_por_split["train"].isdisjoint(ids_por_split["val"])
    assert ids_por_split["train"].isdisjoint(ids_por_split["test"])
    assert ids_por_split["val"].isdisjoint(ids_por_split["test"])

    return splits


def build_transform(split):
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET),
        ]
    )


class SurFakeRGBVideoDataset(Dataset):
    def __init__(self, videos, transform=None, porcentaje_datos=100, reducir=False):
        self.videos = list(videos)
        self.transform = transform
        self.samples = []

        for video in self.videos:
            imagenes = _imagenes_de_video(video.ruta)
            if reducir:
                imagenes = _seleccionar_uniforme(imagenes, porcentaje_datos)
            for img_path in imagenes:
                self.samples.append(
                    {
                        "path": img_path,
                        "label": video.etiqueta,
                        "tipo": video.tipo,
                        "video_id": video.video_id,
                    }
                )

        if not self.samples:
            raise RuntimeError("No se encontraron imagenes para este split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["path"]) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        y = torch.tensor(sample["label"], dtype=torch.long)
        return img, y

    def conteo_clases(self):
        conteo = {0: 0, 1: 0}
        for sample in self.samples:
            conteo[sample["label"]] += 1
        return conteo

    def conteo_tipos(self):
        conteo = {}
        for sample in self.samples:
            conteo[sample["tipo"]] = conteo.get(sample["tipo"], 0) + 1
        return conteo


def construir_sampler_balanceado(dataset):
    conteo = dataset.conteo_clases()
    pesos_clase = {
        etiqueta: 1.0 / max(cantidad, 1)
        for etiqueta, cantidad in conteo.items()
    }
    pesos = [pesos_clase[sample["label"]] for sample in dataset.samples]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(pesos),
        num_samples=len(pesos),
        replacement=True,
    )


def construir_dataloaders(
    root_dir,
    batch_size=16,
    num_workers=4,
    seed=42,
    balancear_train=True,
    porcentaje_datos=100,
):
    splits_video = construir_splits_por_video(root_dir, seed=seed)
    datasets = {
        split: SurFakeRGBVideoDataset(
            videos,
            transform=build_transform(split),
            porcentaje_datos=porcentaje_datos,
            reducir=split == "train",
        )
        for split, videos in splits_video.items()
    }

    sampler_train = (
        construir_sampler_balanceado(datasets["train"])
        if balancear_train
        else None
    )

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=sampler_train is None,
            sampler=sampler_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return datasets, loaders, splits_video


def imprimir_resumen_splits(datasets, splits_video):
    for split in ("train", "val", "test"):
        videos = splits_video[split]
        print(f"\n{split}")
        print("  videos:", len(videos))
        print("  imagenes:", len(datasets[split]))
        print("  clases:", datasets[split].conteo_clases())
        print("  tipos:", datasets[split].conteo_tipos())
