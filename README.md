# MaskedAutoencoder-Jax

## Introduction

This project aims to re-implement [MaskedAutoencoder (CVPR 2022, He *et al.*)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf) using Jax/Flax and running on TPUs. [Publicly released implementation of MAE](https://github.com/facebookresearch/mae) is based on PyTorch+GPU, **whereas the paper's official results are reported based on TensorFlow+TPU**. Hence this project aims to provide an alternative codebase for training a variant of MAE on TPUs.

### MAE Linear Probing Reproduction on ImageNet-1K
We have trained MAEs based on the paper's recipes. Experiments were done on a `v4-64` or `v4-32` TPU pod slice.

| Encoder | Data | Resolution | Epochs | Reimpl. | Original | Config | Wandb | Model |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ViT-B/16 | in1k | 224 | 1600 | 68.1 | 68.0 | [config]() | [log]() | [ckpt]() |
| ViT-L/16 | in1k | 224 | 800 | 73.0 | 73.5 | [config]() | [log]() | [ckpt]() |


## Getting Started

### Environment Setup

To begin, create a TPU instance for training ViTs. We have tested on `v3-8`, `v4-32`  and `v4-64`. We recommend using the `v4-64` pod slice. If you do not have any TPU quota, visit [this link](https://sites.research.google/trc/about/) and apply for the TRC program. **Please adjust the zone according to the email you received from the TRC program.**

```bash
$ gcloud compute tpus tpu-vm create tpu-name \
    --zone=us-central2-b \
    --accelerator-type=v4-64 \
    --version=tpu-ubuntu2204-base 
```

Once the TPU instance is created, clone this repository and install the required dependencies. All dependencies and installation steps are sepcified in the [scripts/setup.sh](./scripts/setup.sh) file. Note that you should use the `gcloud` command to execute the same command on all nodes simultaneously. The `v4-64` pod slice contains 8 computing nodes, each with 4 v4 chips.

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="git clone https://github.com/KAIST-AILab/MaskedAutoencoder-Jax"
```

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="bash MaskedAutoencoder-Jax/scripts/setup.sh"
```

Additionally, log in to your wandb account using the command below. Replace `$WANDB_API_KEY` with your own API key.

```bash
$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; wandb login $WANDB_API_KEY"
```

### Prepare Dataset Shards

`MaskedAutoencoder-Jax` utilizes [webdataset](https://github.com/webdataset/webdataset) to load training samples from various sources, such as huggingface hub and GCS. [Timm](https://github.com/huggingface/pytorch-image-models) provides webdataset versions of [ImageNet-1k](https://huggingface.co/datasets/timm/imagenet-1k-wds) on the huggingface hub. We recommend copying the resources to your GCS bucket for faster download speeds. To download both datasets to your bucket, use the following command:

```bash
$ export HF_TOKEN=...
$ export GCS_DATASET_DIR=gs://...

$ bash scripts/prepare-imagenet1k-dataset.sh
```

For example, you can list the tarfiles in your bucket like this:

```bash
$ gsutil ls gs://mae-storage/datasets/imagenet-1k-wds/
gs://mae-storage/datasets/imagenet-1k-wds/imagenet1k-train-0000.tar
gs://mae-storage/datasets/imagenet-1k-wds/imagenet1k-train-0001.tar
gs://mae-storage/datasets/imagenet-1k-wds/imagenet1k-train-0002.tar
gs://mae-storage/datasets/imagenet-1k-wds/imagenet1k-train-0003.tar
gs://mae-storage/datasets/imagenet-1k-wds/imagenet1k-train-0004.tar
...
```

However, GCS is not the only way to use webdataset. Instead of prefetching into your own bucket, it is also possible to directly stream from the huggingface hub while training.

```bash
$ export TRAIN_SHARDS=https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{0000..1023}.tar
$ export VALID_SHARDS=https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{00..63}.tar

$ python3 src/main.py \
    --train-dataset-shards "pipe:curl -s -L $TRAIN_SHARDS -H 'Authorization:Bearer $HF_TOKEN'" \
    --valid-dataset-shards "pipe:curl -s -L $VALID_SHARDS -H 'Authorization:Bearer $HF_TOKEN'" \
    ...
```
Since intermittent decreases in download performance may occur when streaming from the huggingface hub, we recommend using the GCS bucket for stable download speed and consistent training.

### Train MAEs

You can now pretrain your MAEs using the command below. Replace `$CONFIG_FILE` with the path to the configuration file you want to use. Instead, you can customize your own training recipes by adjusting the [hyperparameters](#hyperparameters). The pretraining presets are available in the [config](./config) folder.

```bash
$ export GCS_DATASET_DIR=gs://...

$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; cd MaskedAutoencoder-Jax; screen -dmL bash $CONFIG_FILE"
```

State the sharded dataset directory in `$GCS_DATASET_DIR`. The training results will be saved to `$GCS_DATASET_DIR/CKPT`. You can specify a local directory path instead of a GCS path to save models locally. If you want to use the pretrained model, you can specify the path to the pretrained model by setting `$GCS_MODEL_PATH`.

**Pretraining Script Example**

```bash
$ export GCS_DATASET_DIR=gs://...

$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; cd MaskedAutoencoder-Jax; screen -dmL bash ./config/pretrain/pretrain-vit-l16-224-in1k-800ep.sh"
```

**Linear Probing Script Example**

You can use either SGD or LARS optimizer for linear probing. Linear probing with LARS optimizer follows the paper's recipe, whereas linear probing with SGD optimizer follows [MoCo v3](https://arxiv.org/abs/2104.02057)'s recipe. Metrics may vary slightly but as long as batch normalization is used, the results should be similar.

```bash
$ export GCS_DATASET_DIR=gs://...
$ export GCS_MODEL_PATH=gs://...

$ gcloud compute tpus tpu-vm ssh tpu-name \
    --zone=us-central2-b \
    --worker=all \
    --command="source ~/miniconda3/bin/activate base; cd MaskedAutoencoder-Jax; screen -dmL bash ./config/linear_probe/ln-lars-vit-l16-224-in1k.sh"
```

### Convert Checkpoints to Timm

To use the pretrained checkpoints, you can convert `.msgpack` to timm-compatible `.pth` files.
```bash
$ python scripts/convert_flax_to_pytorch.py pretrain-vit-l16-224-in1k-800ep-best.msgpack
$ ls pretrain-vit-l16-224-in1k-800ep-best.msgpack  pretrain-vit-l16-224-in1k-800ep-best.pth
```

After converting `.msgpack` to `.pth`, you can load it with timm:
```python
>>> import torch
>>> import timm
>>> model = timm.create_model("vit_large_patch16_224", init_values=1e-4)
>>> model.load_state_dict(torch.load("pretrain-vit-l16-224-in1k-800ep-best.pth"))
<All keys matched successfully>
```

## Hyperparameters

### MaskedAutoencoder
* `--mode`: Training mode of MaskedAutoencoder. Choose `pretrain` for pretraining, `linear` for linear probing, and `finetune` for finetuning.
* `--image_mask_ratio`: Ratio of masked pixels in the input image.

### Image Augmentations
* `--random-crop`: Type of random cropping. Choose `none` for nothing, `rrc` for RandomResizedCrop, and `src` for SimpleResizedCrop proposed in DeiT-III.
* `--color-jitter`: Factor for color jitter augmentation.
* `--auto-augment`: Name of auto-augment policy used in Timm (e.g. `rand-m9-mstd0.5-inc1`).
* `--random-erasing`: Probability of random erasing augmentation.
* `--augment-repeats`: Number of augmentation repetitions.
* `--test-crop-ratio`: Center crop ratio for test preprocessing.
* `--mixup`: Factor (alpha) for Mixup augmentation. Disable by setting to 0.
* `--cutmix`: Factor (alpha) for CutMix augmentation. Disable by setting to 0.
* `--criterion`: Type of classification loss. Choose `ce` for softmax cross entropy and `bce` for sigmoid cross entropy.
* `--label-smoothing`: Factor for label smoothing.

### ViT Architecture
* `--layers`: Number of layers.
* `--dim`: Number of hidden features.
* `--heads`: Number of attention heads.
* `--labels`: Number of classification labels.
* `--layerscale`: Flag to enable LayerScale.
* `--patch-size`: Patch size in ViT embedding layer.
* `--image-size`: Input image size.
* `--posemb`: Type of positional embeddings in ViT. Choose `learnable` for learnable parameters and `sincos2d` for sinusoidal encoding.
* `--pooling`: Type of pooling strategy. Choose `cls` for using `[CLS]` token and `gap` for global average pooling.
* `--dropout`: Dropout rate.
* `--droppath`: DropPath rate (stochastic depth).
* `--grad-ckpt`: Flag to enable gradient checkpointing for reducing memory footprint.

### MAE Decoder Architecture
* `--dec-layers`: Number of decoder layers.
* `--dec-dim`: Number of hidden features in the decoder.
* `--dec-heads`: Number of attention heads in the decoder.
* `--dec-layerscale`: Flag to enable LayerScale in the decoder.
* `--dec-posemb`: Type of positional embeddings in the decoder. Choose `learnable` for learnable parameters and `sincos2d` for sinusoidal encoding.
* `--dec-dropout`: Dropout rate in the decoder.
* `--dec-droppath`: DropPath rate (stochastic depth) in the decoder.
* `--norm-pix-loss`: Flag to normalize pixel loss by the number of pixels.

### Optimization
* `--optimizer`: Type of optimizer. Choose `adamw` for AdamW, `lamb` for LAMB, `sgd` for SGD, and `lars` for LARS.
* `--learning-rate`: Peak learning rate.
* `--weight-decay`: Decoupled weight decay rate.
* `--adam-b1`: Adam beta1.
* `--adam-b2`: Adam beta2.
* `--adam-eps`: Adam epsilon.
* `--lr-decay`: Layerwise learning rate decay rate.
* `--clip-grad`: Maximum gradient norm.
* `--grad-accum`: Number of gradient accumulation steps.
* `--warmup-steps`: Number of learning rate warmup steps.
* `--training-steps`: Number of total training steps.
* `--log-interval`: Number of logging intervals.
* `--eval-interval`: Number of evaluation intervals.

### Random Seeds
* `--init-seed`: Random seed for weight initialization.
* `--mixup-seed`: Random seed for Mixup and CutMix augmentations.
* `--dropout-seed`: Random seed for Dropout regularization.
* `--shuffle-seed`: Random seed for dataset shuffling.
* `--pretrained-ckpt`: Pretrained model path to load from.
* `--label-mapping`: Label mapping file to reuse the pretrained classification head for transfer learning.
* `--noise-seed`: Random seed for input patch masking.

# License

This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.

# Acknowledgement
Thanks to the [TPU Research Cloud](https://sites.research.google/trc/about/) program for providing resources. Models are trained on the TPU `v4-64` or TPU `v4-32` pod slice.

```
@inproceedings{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={16000--16009},
  year={2022}
}
```