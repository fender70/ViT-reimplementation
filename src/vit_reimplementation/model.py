"""Vision Transformer modules built from scratch for learning and experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        channels: int,
        dim: int,
    ) -> None:
        super().__init__()

        if len(image_size) != 2:
            raise ValueError(
                f"Expected 2 dimension image_size but got {len(image_size)}"
            )
        if len(patch_size) != 2:
            raise ValueError(
                f"Expected 2 dimension patch_size but got {len(patch_size)}"
            )
        if not all(x > 0 for x in image_size):
            raise ValueError(
                f"Expected image_size with positive dimensions but got {image_size}"
            )
        if not all(x > 0 for x in patch_size):
            raise ValueError(
                f"Expected patch_size positive dimensions but got {patch_size}"
            )
        if ((image_size[0] % patch_size[0]) != 0) or ((image_size[1] % patch_size[1]) != 0):
            raise ValueError(
                f"Expected image_size divisible by patch_size but got image_size {image_size} and patch_size {patch_size}"
            )
        if not channels > 0:
            raise ValueError(
                f"Expected positive number of channels but got {channels}"
            )
        if not dim > 0:
            raise ValueError(
                f"Expected positive number of dims but got {dim}"
            )

        self.channels = channels
        self.dim = dim
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.patch_height = patch_size[0]
        self.patch_width = patch_size[1]
        self.num_patches_h = self.image_height // self.patch_height
        self.num_patches_w = self.image_width // self.patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.patch_height * self.patch_width * channels

        self.project = nn.Linear(
            in_features = self.patch_dim,
            out_features = dim
        )
        
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        patches = self.patchify(images)
        embeddings = self.project(patches)
        
        return embeddings

    def patchify(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Expected 4 dimension images, but got {images.ndim} dimensions"
            )
        if images.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels but got {images.shape[1]} channels"
            )
        if images.shape[2] != self.image_height:
            raise ValueError(
                f"Expected height {self.image_height} but got {images.shape[2]}"
            )
        if images.shape[3] != self.image_width:
            raise ValueError(
                f"Expected width {self.image_width} but got {images.shape[3]}"
            )

        batch_size = images.shape[0]
            
        images = images.reshape(batch_size, self.channels, self.num_patches_h, self.patch_height, self.num_patches_w, self.patch_width,)
        images = images.permute(0, 2, 4, 3, 5, 1)
        images = images.reshape(batch_size, self.num_patches, self.patch_dim)
        
        return images


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim
    ) -> None:
        super().__init__()

        self.dim = dim

        self.linear_layer_1 = nn.Linear(
            dim,
            hidden_dim
        )

        self.gelu_layer = nn.GELU()

        self.linear_layer_2 = nn.Linear(
            hidden_dim,
            dim
        )
        
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected final dimension {self.dim} but got {x.shape[-1]}"
            )
            
        x_l1 = self.linear_layer_1(x)
        x_gelu = self.gelu_layer(x_l1)
        out = self.linear_layer_2(x_gelu)

        return out


class Attention(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        tensors = [query, key, value]

        for tensor in tensors:
            if tensor.ndim!= 4:
                raise ValueError(
                    f"Expected tensor rank 4, but got {tensor.ndim}"
                )
        if not (query.shape[:2] == key.shape[:2] == value.shape[:2]):
            raise ValueError(
                "Expected query, key, and value to have matching batch size "
                f"and heads, but got query: {query.shape}, key: {key.shape}, "
                f"value: {value.shape}"
            )
        if query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "Expected query and key to have matching head dimensions, "
                f"but got query head dimensions: {query.shape[-1]}, "
                f"key head dimensions: {key.shape[-1]}"
            )
        if key.shape[2] != value.shape[2]:
            raise ValueError(
                "Expected key and value to have matching sequence lengths, "
                f"but got key sequence length: {key.shape[2]}, "
                f"value sequence length: {value.shape[2]}"
            )

        head_dim = query.shape[-1]

        scores = (
            query @ key.transpose(-2, -1)
        ) / (head_dim ** 0.5)

        weights = F.softmax(scores, dim=-1)
        output = weights @ value

        return output, weights

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if not d_model > 0:
            raise ValueError(
                f"Expected positive d_model but got {d_model}"
            )
        if not num_heads > 0:
            raise ValueError(
                f"Expected positive num_heads but got {num_heads}"
            )
        if not ((d_model % num_heads) == 0):
            raise ValueError(
                f"Expected num_heads to divide d_model but got d_model {d_model}, and num_heads {num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.project_query = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.project_key = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.project_value = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.project_output = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )

        self.attention = Attention()

    def split_heads(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected rank-3 tensor, but got {tensor.ndim}"
            )
        if tensor.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected final dimension {self.d_model}, but got {tensor.shape[-1]}"
            )
        
        batch_size, sequence_length, _ = tensor.shape
        
        out = tensor.reshape(batch_size, sequence_length, self.num_heads, self.d_head)

        return out.transpose(1, 2)

    def combine_heads(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:

        if tensor.ndim != 4:
            raise ValueError(
                f"Expected rank-4 tensor, but got {tensor.ndim}"
            )
        if tensor.shape[1] != self.num_heads:
            raise ValueError(
                f"Expected num_heads {self.num_heads}, but got {tensor.shape[1]}"
            )
        if tensor.shape[-1] != self.d_head:
            raise ValueError(
                f"Expected head dimensions {self.d_head}, but got {tensor.shape[-1]}"
            )
            
        tensor = tensor.transpose(1, 2)

        batch_size, sequence_length, _, _ = tensor.shape

        out = tensor.reshape(batch_size, sequence_length, -1)

        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        query, key, value = self.project_query(query), self.project_key(key), self.project_value(value)
        query, key, value = self.split_heads(query), self.split_heads(key), self.split_heads(value)

        output, weights = self.attention(
            query=query,
            key=key,
            value=value
        )

        output = self.combine_heads(output)
        output = self.project_output(output)

        return output, weights


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.layernorm1 = nn.LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, num_heads)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Self Attention
        x_n = self.layernorm1(x)
        x_p, weights = self.attention(
            query=x_n,
            key=x_n,
            value=x_n)
        x = x + x_p

        # Feed Forward
        x_n2 = self.layernorm2(x)
        x_p2 = self.ffn(x_n2)
        output = x + x_p2

        return output, weights


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_heads, hidden_dim,))

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        all_weights = []

        output = x
        
        for layer in self.layers:
            output, weights = layer(output)
            all_weights.append(weights)

        return output, all_weights


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        channels,
        dim,
        num_layers,
        num_heads,
        hidden_dim,
        num_classes,
    ) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            channels,
            dim
        )
        
        self.num_patches = self.patch_embedding.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.pos_embedding = nn.Embedding(
            self.num_patches + 1,
            dim,
        )

        self.transformer = Transformer(
            num_layers=num_layers,
            d_model=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
        )

        self.project_classes = nn.Linear(
            in_features=dim,
            out_features=num_classes,
        )

        self.layernorm_final = nn.LayerNorm(dim)

    def forward(
        self,
        images: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        x_emb = self.patch_embedding(images)
        
        batch_size = images.shape[0]
    
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x_concat = torch.cat([cls_tokens, x_emb], dim=1)

        positions = torch.arange(start=0, end=self.num_patches + 1, device=x_emb.device).unsqueeze(0)

        x = x_concat + self.pos_embedding(positions)

        x_final, all_weights = self.transformer(x)

        x_final_norm = self.layernorm_final(x_final)

        logits = self.project_classes(x_final_norm[:, 0, :]) # Grabs the CLS token

        return logits, all_weights
