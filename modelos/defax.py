import torch.nn as nn

from .clasificador_mlp import ClasificadorMLP
from .cross_attention_fusion import CrossAttentionFusion
from .efficientnet_tokens import EfficientNetTokens
from .pooling_tokens import PoolingTokens
from .swin_base_tokens import SwinBaseTokens


class DeFaX(nn.Module):
    def __init__(
        self,
        swin_modelo="swin_base_patch4_window7_224",
        efficient_modelo="efficientnet_b0",
        pretrained=True,
        num_heads=8,
        dropout_atencion=0.0,
        dropout_clasificador=0.1,
        num_clases=2,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.swin = SwinBaseTokens(
            nombre_modelo=swin_modelo,
            pretrained=pretrained,
            imprimir_shapes=imprimir_shapes,
        )
        self.efficient = EfficientNetTokens(
            nombre_modelo=efficient_modelo,
            pretrained=pretrained,
            imprimir_shapes=imprimir_shapes,
        )
        self.fusion = CrossAttentionFusion(
            dim_swin=1024,
            dim_eff=320,
            num_heads=num_heads,
            dropout=dropout_atencion,
            imprimir_shapes=imprimir_shapes,
        )
        self.pooling = PoolingTokens(
            dim_tokens=1024,
            imprimir_shapes=imprimir_shapes,
        )
        self.clasificador = ClasificadorMLP(
            dim_entrada=1024,
            num_clases=num_clases,
            dropout=dropout_clasificador,
            imprimir_shapes=imprimir_shapes,
        )

    def forward(self, x):
        tokens_swin = self.swin(x)
        tokens_eff = self.efficient(x)
        tokens_fusion = self.fusion(tokens_swin, tokens_eff)
        z = self.pooling(tokens_fusion)
        logits = self.clasificador(z)
        return logits
