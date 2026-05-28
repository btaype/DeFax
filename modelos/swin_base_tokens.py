import torch.nn as nn
import timm


class SwinBaseTokens(nn.Module):
    def __init__(
        self,
        nombre_modelo="swin_base_patch4_window7_224",
        pretrained=True,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.imprimir_shapes = imprimir_shapes
        self.cs = 1024
        self.backbone = timm.create_model(
            nombre_modelo,
            pretrained=pretrained,
            num_classes=0,
        )

    def forward(self, x):
        if x.dim() != 4 or x.shape[1:] != (3, 224, 224):
            raise ValueError(f"recibido swin: {x.shape}")

        salida = self.backbone.forward_features(x)

        if salida.dim() == 4:
            if salida.shape[-1] == self.cs:
                tokens_swin = salida.flatten(1, 2)
            else:
                tokens_swin = salida.flatten(2).transpose(1, 2)
        elif salida.dim() == 3:
            tokens_swin = salida
        else:
            raise ValueError(f"recibido swin: {salida.shape}")

        if tokens_swin.dim() != 3 or tokens_swin.shape[-1] != self.cs:
            raise ValueError(f"recibido swin: {tokens_swin.shape}")

        if self.imprimir_shapes:
            print("swin tokens:", tokens_swin.shape)

        return tokens_swin
