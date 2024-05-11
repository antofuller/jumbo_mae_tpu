# Copyright 2022 Contributors of https://github.com/young-geng/m3ae_public
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

from typing import Callable, Optional, Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial


def mask_union(mask1, mask2):
    return jnp.logical_or(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_intersection(mask1, mask2):
    return jnp.logical_and(mask1 > 0, mask2 > 0).astype(jnp.float32)


def mask_not(mask):
    return 1.0 - mask


def mask_select(mask, this, other=None):
    if other is None:
        other = jnp.array(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask == 0.0, this, other)


def no_mask(x):
    return jnp.zeros(x.shape[:2])


def all_mask(x):
    return jnp.ones(x.shape[:2])

def patch_mse_loss(patch_output, patch_target, valid=None):
    if valid is None:
        valid = all_mask(patch_target)
    valid_ratio = jnp.sum(valid, axis=-1) / valid.shape[-1]
    return jnp.mean(
        jnp.mean(
            jnp.where(
                valid > 0.0,
                jnp.mean(jnp.square(patch_target - patch_output), axis=-1),
                jnp.array(0.0),
            ),
            axis=-1,
        ) / valid_ratio
    )


def extract_patches(inputs, patch_size):
    batch, height, width, channels = inputs.shape
    height, width = height // patch_size, width // patch_size
    x = jnp.reshape(inputs, (batch, height, patch_size, width, patch_size, channels))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * width, patch_size**2 * channels))
    return x


def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = jnp.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
    x = jnp.swapaxes(x, 2, 3)
    x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
    return x

def index_sequence(x, ids):
    return x[:, ids, ...]


def random_masking(x, rng, keep_len, padding_mask=None):
    batch, length, dim = x.shape
    noise = jax.random.uniform(rng, (length,), dtype=jnp.float32)
    ids_shuffle = jnp.argsort(noise, axis=0)
    ids_restore = jnp.argsort(ids_shuffle, axis=0)
    kept = index_sequence(x, ids_shuffle[:keep_len])
    mask = jnp.ones([batch, length], dtype=jnp.float32)
    mask = mask.at[:, :keep_len].set(0.0)
    mask = index_sequence(mask, ids_restore)

    if padding_mask is None:
        return kept, mask, ids_restore

    padding_mask_kept = index_sequence(padding_mask, ids_shuffle[:keep_len])
    return kept, mask, ids_restore, padding_mask_kept
