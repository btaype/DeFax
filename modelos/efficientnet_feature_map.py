import torch.nn as nn
import timm


class EfficientNetFeatureMap(nn.Module):
    def __init__(
        self,
        nombre_modelo="efficientnet_b0",
        pretrained=True,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.imprimir_shapes = imprimir_shapes
        self.ce = 320
        self.backbone = timm.create_model(
            nombre_modelo,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )

    def forward(self, x):
        if x.dim() != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError(f"recibido: {x.shape}")

        mapa = self.backbone(x)[0]

        if mapa.dim() != 4 or mapa.shape[1] != self.ce:
            raise ValueError(f"recibido: {mapa.shape}")

        if self.imprimir_shapes:
            print("efficient map:", mapa.shape)

        return mapa
