import torch.nn as nn

from .efficientnet_feature_map import EfficientNetFeatureMap


class EfficientNetTokens(nn.Module):
    def __init__(
        self,
        nombre_modelo="efficientnet_b0",
        pretrained=True,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.imprimir_shapes = imprimir_shapes
        self.extractor = EfficientNetFeatureMap(
            nombre_modelo=nombre_modelo,
            pretrained=pretrained,
            imprimir_shapes=imprimir_shapes,
        )

    def forward(self, x):
        mapa = self.extractor(x)
        tokens_eff = mapa.flatten(2).transpose(1, 2)

        if tokens_eff.dim() != 3 or tokens_eff.shape[-1] != self.extractor.ce:
            raise ValueError(f"recibido efficient: {tokens_eff.shape}")

        if self.imprimir_shapes:
            print("efficient tokens:", tokens_eff.shape)

        return tokens_eff
