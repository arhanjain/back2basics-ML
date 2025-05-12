# Imports
import cv2
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import multiprocessing
import wandb
import torch

from functools import partial
from celebrity_dataset import CelebrityDataset, TRAIN_LIST, VAL_LIST, ResizeTransform, NormalizeTransform, UnNormalizeTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

# Hyperparameters
LR = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 64
NUM_WORKERS = 8
LOG_INTERVAL = 100
SEED = 42
PRECISION = jnp.bfloat16
torch.manual_seed(SEED)

class ConvBlock(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs: nnx.Rngs, dtype: jnp.dtype):
        self.conv1 = nnx.Conv(in_ch, out_ch, kernel_size=(3, 3), padding='SAME', rngs=rngs, dtype=dtype)
        self.bn1 = nnx.BatchNorm(out_ch, rngs=rngs, dtype=dtype)
        self.conv2 = nnx.Conv(out_ch, out_ch, kernel_size=(3, 3), padding='SAME', rngs=rngs, dtype=dtype)
        self.bn2 = nnx.BatchNorm(out_ch, rngs=rngs, dtype=dtype)

    def __call__(self, x, *, training: bool):
        x = nnx.relu(self.bn1(self.conv1(x), use_running_average=not training))
        x = nnx.relu(self.bn2(self.conv2(x), use_running_average=not training))
        return x

class Autoencoder(nnx.Module):
    def __init__(self, in_ch, out_ch, rngs: nnx.Rngs, dtype: jnp.dtype):
        conv = partial(ConvBlock, rngs=rngs, dtype=dtype)
        convTranspose = partial(nnx.ConvTranspose, rngs=rngs, dtype=dtype)

        self.down1 = conv(in_ch, 64)
        self.down2 = conv(64, 128) 
        self.down3 = conv(128, 256)
        self.down4 = conv(256, 512) 
        
        self.bottleneck1 = nnx.Linear(14 * 14 * 512, 2048, dtype=dtype, rngs=rngs)
        self.bottleneck2 = nnx.Linear(2048, 14 * 14 * 1024, dtype=dtype, rngs=rngs)
        # self.mid = conv(512, 1024, rngs, dtype)

        # Upsampling path (ConvTranspose + ConvBlock)
        self.upconv4 = convTranspose(1024, 512, kernel_size=(2, 2), strides=(2, 2))
        self.up4 = conv(512, 512)

        self.upconv3 = convTranspose(512, 256, kernel_size=(2, 2), strides=(2, 2))
        self.up3 = conv(256, 256)

        self.upconv2 = convTranspose(256, 128, kernel_size=(2, 2), strides=(2, 2))
        self.up2 = conv(128, 128)

        self.upconv1 = convTranspose(128, 64, kernel_size=(2, 2), strides=(2, 2))
        self.up1 = conv(64, 64)

        self.out_conv = nnx.Conv(64, out_ch, kernel_size=(1, 1), dtype=dtype, rngs=rngs)
        self.pool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x, training: bool):
        d1 = self.down1(x, training=training)  # 224x224x3 -> 224x224x64
        d2 = self.down2(self.pool(d1), training=training)  # 224x224x64 -> 112x112x128
        d3 = self.down3(self.pool(d2), training=training)  # 112x112x128 -> 56x56x256
        d4 = self.down4(self.pool(d3), training=training)  # 56x56x256 -> 28x28x512

        # mid = self.mid(self.pool(d4), training=training) # 28x28x512 -> 14x14x1024
        mid = self.bottleneck1(self.pool(d4).reshape(x.shape[0], -1)) # 14x14x512 -> 2048

        bottleneck_activation = nnx.relu(mid)

        mid = self.bottleneck2(bottleneck_activation) # 2048 -> 14x14x1024
        mid = nnx.relu(mid)
        mid = mid.reshape(x.shape[0], 14, 14, 1024)

        u4 = self.up4(self.upconv4(mid), training=training)
        u3 = self.up3(self.upconv3(u4), training=training)
        u2 = self.up2(self.upconv2(u3), training=training)
        u1 = self.up1(self.upconv1(u2), training=training)
        return self.out_conv(u1), bottleneck_activation

def KLDivergence(p, q):
    return p * jnp.log(p / q) + (1 - p) * jnp.log((1 - p) / (1 - q))

def loss_function(model: Autoencoder, batch):
    logits, bottleneck_activation = model(batch, training=True)
    loss = optax.l2_loss(logits, batch).mean()

    # bottleneck is B x hidden_dim
    p_hat = bottleneck_activation.mean(axis=0)
    eps = jnp.array(1e-5, dtype=jnp.float32)
    sparsity_loss = 0.05 * jnp.sum(KLDivergence(0.05, jax.lax.clamp(eps, p_hat.astype(jnp.float32), 1 - eps))) # clamp to avoid log(0)
    sparsity_loss = sparsity_loss.astype(jnp.bfloat16)
    total_loss = loss + sparsity_loss
    return total_loss, logits

@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_function, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss.astype(jnp.float32), logits=logits)
    optimizer.update(grads=grads)

@nnx.jit
def val_step(model: nnx.Module, batch):
    logits, _ = model(batch, training=False)
    loss = optax.l2_loss(logits, batch).mean()
    return loss, logits

def apply_transforms(transforms, x):
    for transform in transforms:
        x = transform(x)
    return x

# Collate function
def collate_fn(x):
    return np.stack(x)


if __name__ == "__main__":
    wandb.init(
        project="back2basics",
        name="sparse-autoencoder",
        config={
            "lr": LR,
            "momentum": MOMENTUM,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "seed": SEED
        },
        # mode="disabled"
    )

    ds_stats = np.load('data/celeb_faces/dataset_stats.npy', allow_pickle=True).item()
    mean, std = ds_stats['mean'], ds_stats['std']

    transforms = partial(apply_transforms, [ResizeTransform(224), NormalizeTransform(mean, std)])
    train_ds = CelebrityDataset(TRAIN_LIST, transform=transforms)
    val_ds = CelebrityDataset(VAL_LIST, transform=transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        multiprocessing_context=multiprocessing.get_context('spawn') if NUM_WORKERS > 0 else None,
        # persistent_workers=NUM_WORKERS > 0,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        multiprocessing_context=multiprocessing.get_context('spawn') if NUM_WORKERS > 0 else None,
    )

    model = Autoencoder(3, 3, rngs=nnx.Rngs(42), dtype=PRECISION)
    optimizer = nnx.Optimizer(model, tx=optax.adamw(LR, MOMENTUM))
    train_metrics = nnx.MultiMetric(
        loss = nnx.metrics.Average("loss")
    )
    
    val_batch = next(iter(val_loader))
    val_batch = jnp.array(val_batch)
    for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch = jnp.array(batch).astype(PRECISION)
        train_step(model, optimizer, train_metrics, batch)
        if step % LOG_INTERVAL == 0:
            all_metrics = train_metrics.compute()
            train_metrics.reset()

            _, logits = val_step(model, val_batch)
            inputs = jax.device_get(val_batch)
            outputs = jax.device_get(logits)

            sampled_indices = np.random.choice(outputs.shape[0], size=3, replace=False)
            sampled_inputs = inputs[sampled_indices]
            sampled_outputs = outputs[sampled_indices]

            unnormalized_inputs = UnNormalizeTransform.from_stats(mean, std, sampled_inputs)
            unnormalized_inputs = np.concatenate(unnormalized_inputs, axis=1)
            unnormalized_outputs = UnNormalizeTransform.from_stats(mean, std, sampled_outputs)
            unnormalized_outputs = np.concatenate(unnormalized_outputs, axis=1)

            examples = np.concatenate([unnormalized_inputs, unnormalized_outputs], axis=0).astype(np.float32)
            all_metrics["examples"] = wandb.Image(examples, caption="Input and Output")

            wandb.log(all_metrics, step=step)

