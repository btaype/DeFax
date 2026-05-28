import torch.nn as nn


class PoolingTokens(nn.Module):
    def __init__(self, dim_tokens=1024, imprimir_shapes=False):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.imprimir_shapes = imprimir_shapes

    def forward(self, tokens):
        if tokens.dim() != 3 or tokens.shape[-1] != self.dim_tokens:
            raise ValueError(f"recibido pooling: {tokens.shape}")

        z = tokens.mean(dim=1)

        if z.dim() != 2 or z.shape[-1] != self.dim_tokens:
            raise ValueError(f"recibido pooling: {z.shape}")

        if self.imprimir_shapes:
            print("pooling:", z.shape)

        return z
