import torch.nn as nn


class ClasificadorMLP(nn.Module):
    def __init__(
        self,
        dim_entrada=1024,
        num_clases=2,
        dropout=0.1,
        imprimir_shapes=False,
    ):
        super().__init__()
        self.dim_entrada = dim_entrada
        self.num_clases = num_clases
        self.imprimir_shapes = imprimir_shapes
        self.mlp = nn.Sequential(
            nn.Linear(dim_entrada, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_clases),
        )

    def forward(self, z):
        if z.dim() != 2 or z.shape[-1] != self.dim_entrada:
            raise ValueError(f"recibido mlp: {z.shape}")

        logits = self.mlp(z)

        if logits.dim() != 2 or logits.shape[-1] != self.num_clases:
            raise ValueError(f"recibido mlp: {logits.shape}")

        if self.imprimir_shapes:
            print("logits:", logits.shape)

        return logits
