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

import argparse
from functools import partial
from typing import Callable

import flax
import flax.linen as nn
import flax.linen.initializers as init

import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

from dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from modeling import ViT, MAEDecoder
from utils import get_layer_index_fn, modified_lamb
from utils_mae import extract_patches, index_sequence, patch_mse_loss

Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))

CRITERION_COLLECTION = {
    "ce": optax.softmax_cross_entropy,
    "bce": lambda x, y: optax.sigmoid_binary_cross_entropy(x, y > 0).mean(-1),
}
OPTIMIZER_COLLECTION = {
    "adamw": optax.adamw,
    "lamb": modified_lamb,
}


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey
    noise_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        mixup_rng, new_mixup_rng = jax.random.split(self.mixup_rng)
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        noise_rng, new_noise_rng = jax.random.split(self.noise_rng)

        rngs = {"mixup": mixup_rng, "dropout": dropout_rng, "noise": noise_rng}
        updates = {"mixup_rng": new_mixup_rng, "dropout_rng": new_dropout_rng, "noise_rng": new_noise_rng}
        return rngs, updates

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
            noise_rng=shard_prng_key(self.noise_rng),
        )


class PretrainModule(nn.Module):
    model: ViT
    decoder_model: MAEDecoder
    image_size: int
    norm_pix_loss: bool = False

    def setup(self):
        self.image_mask_embedding = self.param("image_mask_embedding", init.truncated_normal(0.02), (1, 1, self.decoder_model.dec_dim))
        self.decoder_proj = Dense(self.decoder_model.dec_dim)
        self.decoder_image_output = Dense(self.model.patch_size ** 2 * 3)

    def __call__(self, images: Array, det: bool = True) -> ArrayTree:
        # Normalize the pixel values in TPU devices, instead of copying the normalized
        # float values from CPU. This may reduce both memory usage and latency.
        images = jnp.moveaxis(images, 1, 3).astype(jnp.float32) / 0xFF
        images = (images - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        x, image_mask, image_ids_restore = self.model(images, det=False) # No deterministic mode due to random masking

        # DECODER: linear project and separate cls token (w/o CLS)
        x = self.decoder_proj(x)
        encoder_cls, image_x = x[:, :3, :], x[:, 3:, :] # x[:, 0, :] decreases dimension
        bs, seq_len, dim = image_x.shape
        
        # Mask the input representation
        masked_image_x = jnp.broadcast_to(
            self.image_mask_embedding, (bs, int(image_ids_restore.shape[0] * self.model.image_mask_ratio), self.decoder_model.dec_dim),
        )
        image_x = index_sequence(jnp.concatenate([image_x, masked_image_x], axis=1), image_ids_restore)

        # Feedforward the decoder (w/ CLS)
        x = jnp.concatenate([encoder_cls, image_x], axis=1)
        x = self.decoder_model(x, det=det)

        # Reconstruct the output based on image patches only (w/o CLS)
        image_x = x[:, 3:, :]
        image_output = self.decoder_image_output(image_x)

        image_patches = extract_patches(images, self.model.patch_size)
        
        if self.norm_pix_loss:
            mean = jnp.mean(image_patches, axis=-1, keepdims=True)
            var = jnp.var(image_patches, axis=-1, keepdims=True)
            image_patches = (image_patches - mean) / jnp.sqrt(var + 1e-6)

        loss = patch_mse_loss(image_output, image_patches, image_mask)

        return {"loss": loss}
        

@partial(jax.pmap, axis_name="batch", donate_argnums=0)
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    def loss_fn(params: ArrayTree) -> ArrayTree:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs)
        metrics = jax.tree_map(jnp.mean, metrics)
        return metrics["loss"], metrics

    def update_fn(state: TrainState) -> TrainState:
        # Collect a global gradient from the accumulated gradients and apply actual
        # parameter update with resetting the accumulations to zero.
        grads = jax.tree_map(lambda g: g / state.micro_in_mini, state.grad_accum)
        return state.apply_gradients(
            grads=jax.lax.pmean(grads, axis_name="batch"),
            grad_accum=jax.tree_map(jnp.zeros_like, state.grad_accum),
            micro_step=state.micro_step % state.micro_in_mini,
        )

    rngs, updates = state.split_rngs()
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Update parameters with the gradients. If the gradient accumulation is enabled,
    # then the parameters will be updated at the end of each mini-batch step. In every
    # micro steps, the gradients will be accumulated.
    if state.grad_accum is None:
        state = state.apply_gradients(grads=jax.lax.pmean(grads, axis_name="batch"))
    else:
        state = state.replace(
            grad_accum=jax.tree_map(lambda ga, g: ga + g, state.grad_accum, grads),
            micro_step=state.micro_step + 1,
        )
        state = jax.lax.cond(
            state.micro_step == state.micro_in_mini, update_fn, lambda x: x, state
        )
    return state.replace(**updates), metrics | state.opt_state.hyperparams


@partial(jax.pmap, axis_name="batch")
def validation_step(state: TrainState, batch: ArrayTree) -> ArrayTree:
    rngs, _ = state.split_rngs()
    metrics = state.apply_fn({"params": state.params}, images=batch[0], det=False, rngs=rngs)
    metrics["num_samples"] = batch[0].shape[0]
    return jax.lax.psum(metrics, axis_name="batch")


def create_train_state(args: argparse.Namespace) -> TrainState:
    model = ViT(
        layers=args.layers,
        dim=args.dim,
        heads=args.heads,
        labels=args.labels,
        layerscale=args.layerscale,
        patch_size=args.patch_size,
        image_size=args.image_size,
        posemb=args.posemb,
        pooling=args.pooling,
        dropout=args.dropout,
        droppath=args.droppath,
        grad_ckpt=args.grad_ckpt,
        image_mask_ratio=args.image_mask_ratio,
        linear_probing=False,
    )

    decoder_model = MAEDecoder(
        dec_layers=args.dec_layers,
        dec_dim=args.dec_dim,
        dec_heads=args.dec_heads,
        dec_layerscale=args.dec_layerscale,
        dec_posemb=args.dec_posemb,
        dec_dropout=args.dec_dropout,
        dec_droppath=args.dec_droppath,
        grad_ckpt=args.grad_ckpt,
        patch_size=args.patch_size,
        image_size=args.image_size,
    )
    module = PretrainModule(
        model=model,
        decoder_model=decoder_model,
        image_size=args.image_size,
        norm_pix_loss=args.norm_pix_loss,
    )

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((1, 3, args.image_size, args.image_size), dtype=jnp.uint8),
    }
    init_rngs = {"params": jax.random.PRNGKey(args.init_seed)}
    print(module.tabulate(init_rngs, **example_inputs))

    params = module.init(init_rngs, **example_inputs)["params"]

    if args.grad_accum > 1:
        grad_accum = jax.tree_map(jnp.zeros_like, params)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
        learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[args.optimizer](
            learning_rate=learning_rate,
            b1=args.adam_b1,
            b2=args.adam_b2,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        if args.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(args.lr_decay ** (args.layers - i))
                for i in range(args.layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=args.layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if args.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/PRETRAIN.md
    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py#L40-L41
    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py#L156-L184
    lr_peak_value = args.learning_rate * args.train_batch_size / 256
    print(f"Peak learning rate: {lr_peak_value:.1e}")
    
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=lr_peak_value,
        warmup_steps=args.warmup_steps,
        decay_steps=args.training_steps,
        end_value=1e-5,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed + jax.process_index()),
        noise_rng=jax.random.PRNGKey(args.noise_seed + jax.process_index()),
        micro_step=0,
        micro_in_mini=args.grad_accum,
        grad_accum=grad_accum if args.grad_accum > 1 else None,
    )
