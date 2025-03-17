# Copyright 2024 Jungwoo Park (affjljoo3581) and Young Jin Ahn (snoop2head)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from chex import Array

from utils import fixed_sincos2d_embeddings
from utils_mae import random_masking # patch related

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "cls"

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    
    # MAE related
    image_mask_ratio: float = 0.75
    linear_probing: bool = False
    batch_norm: bool = False

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2

@dataclass
class MAEDecoderBase:
    dec_layers: int = 6
    dec_dim: int = 512
    dec_heads: int = 8
    dec_layerscale: bool = False

    dec_posemb: Literal["learnable", "sincos2d"] = "learnable"

    dec_dropout: float = 0.0
    dec_droppath: float = 0.0
    grad_ckpt: bool = False

    patch_size: int = 16
    image_size: int = 224

    @property
    def kwargs(self) -> dict[str, Any]:
        decoder_kwargs = {f.name: getattr(self, f.name) for f in fields(MAEDecoderBase)}
        return {k.replace("dec_", ""): v for k, v in decoder_kwargs.items()} # replace "dec_" with "" for the decoder kwargs
    
    @property
    def head_dim(self) -> int:
        return self.dec_dim // self.dec_heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dec_dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2

class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        return x


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        z = jnp.einsum("bqhd,bkhd->bhqk", self.wq(x) / self.head_dim**0.5, self.wk(x))
        z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
        return self.drop(self.wo(z), det)


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)

class ViTLayer(ViTBase, nn.Module):
    jumbo_mlp: nn.Module = None  # You must provide it explicitly later
    num_cls_tokens: int = 3

    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

        self.drop_path1 = nn.Dropout(self.droppath, broadcast_dims=(1, 2))
        self.drop_path2 = nn.Dropout(self.droppath, broadcast_dims=(1, 2))
        self.drop_path3 = nn.Dropout(self.droppath, broadcast_dims=(1,))

        self.scale1 = self.scale2 = self.scale3 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))
            self.scale3 = self.param("scale3", init.constant(1e-4), (self.dim * self.num_cls_tokens,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = x + self.drop_path1(self.scale1 * self.attn(self.norm1(x), det), det)

        cls_tokens, patch_tokens = x[:, :self.num_cls_tokens, :], x[:, self.num_cls_tokens:, :]

        bs = cls_tokens.shape[0]
        cls_tokens_concat = cls_tokens.reshape(bs, -1)
        cls_tokens_concat = self.norm3(cls_tokens_concat)
        cls_tokens_concat = cls_tokens_concat + self.drop_path3(self.scale3 * self.jumbo_mlp(cls_tokens_concat, det), det)

        patch_tokens = patch_tokens + self.drop_path2(self.scale2 * self.ff(self.norm2(patch_tokens), det), det)

        cls_tokens = cls_tokens_concat.reshape(bs, self.num_cls_tokens, -1)
        x = jnp.concatenate([cls_tokens, patch_tokens], axis=1)

        return x


class LinearCLS(nn.Module):
    labels: int
    batch_norm: bool
    dim: int

    @nn.compact
    def __call__(self, x: Array, det: bool) -> Array:
        if self.batch_norm:
            x = nn.BatchNorm(use_running_average=det, axis_name="batch")(x)
        x = nn.Dense(self.labels, kernel_init=init.truncated_normal(0.02), use_bias=True)(x)
        return x

class ViT(ViTBase, nn.Module):
    num_cls_tokens: int = 3

    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        jumbo_kwargs = dict(self.kwargs)
        jumbo_kwargs['hidden_dim'] = self.hidden_dim * self.num_cls_tokens
        jumbo_kwargs['dim'] = self.dim * self.num_cls_tokens
        self.jumbo_mlp = FeedForward(**jumbo_kwargs)

        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [
            layer_fn(jumbo_mlp=self.jumbo_mlp, num_cls_tokens=self.num_cls_tokens, **self.kwargs)
            for _ in range(self.layers)
        ]

        self.norm = nn.LayerNorm()
        self.head = (
            LinearCLS(self.labels, self.batch_norm, dim=self.dim * self.num_cls_tokens)
            if self.labels > 0 else None
        )
        self.cls_tokens = self.param("cls_tokens", init.zeros, (1, self.num_cls_tokens, self.dim))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = self.embed(x)
        bs = x.shape[0]

        cls_tokens = jnp.tile(self.cls_tokens, (bs, 1, 1))
        x = jnp.concatenate((cls_tokens, x), axis=1)

        if self.head is None:  # MAE mode
            cls_tokens, patch_tokens = x[:, :self.num_cls_tokens, :], x[:, self.num_cls_tokens:, :]
            seq_len = patch_tokens.shape[1]
            image_keep_length = int(seq_len * (1.0 - self.image_mask_ratio))
            patch_tokens, image_mask, image_ids_restore = random_masking(patch_tokens, self.make_rng("noise"), image_keep_length)
            x = jnp.concatenate((cls_tokens, patch_tokens), axis=1)

        x = self.drop(x, det)

        for layer in self.layer:
            x = layer(x, det)

        x = self.norm(x)

        if self.head is None:  # MAE mode
            return x, image_mask, image_ids_restore

        if self.linear_probing:
            x = jax.lax.stop_gradient(x)

        x = x[:, :self.num_cls_tokens, :].reshape(bs, -1)  # concatenate CLS tokens
        
        return self.head(x, det)

class MAEDecoder(MAEDecoderBase, nn.Module):
    num_cls_tokens: int = 3

    def setup(self):
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.dec_layer = [layer_fn(**self.kwargs) for _ in range(self.dec_layers)]
        self.dec_norm = nn.LayerNorm()

        self.dec_wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dec_dim)

    def __call__(self, x: Array, det: bool = True) -> Array:
        bs = x.shape[0]

        cls_tokens, patch_tokens = x[:, :self.num_cls_tokens, :], x[:, self.num_cls_tokens:, :]
        patch_tokens = patch_tokens + self.dec_wpe.reshape(1, -1, patch_tokens.shape[-1])

        x = jnp.concatenate((cls_tokens, patch_tokens), axis=1)

        for layer in self.dec_layer:
            x = layer(x, det)

        x = self.dec_norm(x)
        return x
