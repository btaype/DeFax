import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim_swin=1024,
        dim_eff=320,
        num_heads=8,
        dropout=0.0,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.imprimir_shapes = imprimir_shapes
        self.dim_swin = dim_swin
        self.dim_eff = dim_eff
        self.atencion = nn.MultiheadAttention(
            embed_dim=dim_swin,
            num_heads=num_heads,
            kdim=dim_eff,
            vdim=dim_eff,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens_swin, tokens_eff):
        if tokens_swin.dim() != 3 or tokens_swin.shape[-1] != self.dim_swin:
            raise ValueError(f"recibido cross attention: {tokens_swin.shape}")

        if tokens_eff.dim() != 3 or tokens_eff.shape[-1] != self.dim_eff:
            raise ValueError(f"recibido cross attention: {tokens_eff.shape}")

        if tokens_swin.shape[0] != tokens_eff.shape[0]:
            raise ValueError(f"recibido cross atention: {tokens_swin.shape[0]} vs {tokens_eff.shape[0]}")

        q = tokens_swin
        k = tokens_eff
        v = tokens_eff

        fusion, _ = self.atencion(
            query=q,
            key=k,
            value=v,
            need_weights=False,
        )

        if fusion.shape != tokens_swin.shape:
            raise ValueError(f"recibido cross atention: {fusion.shape}")

        if self.imprimir_shapes:
            print("cross attention:", fusion.shape)

        return fusion
