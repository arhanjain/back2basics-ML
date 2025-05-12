import tyro
import wandb
import jax
import optax
import jax.numpy as jnp
import torch
import numpy as np
import multiprocessing
torch.manual_seed(42)

from einops import rearrange    
from dataclasses import dataclass
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from flax import nnx    
from celebrity_dataset import CelebrityDataset, TRAIN_LIST, VAL_LIST, ResizeTransform, NormalizeTransform, UnNormalizeTransform

class VAE(nnx.Module):
    def __init__(self, 
                 height, 
                 width, 
                 channels, 
                 rngs: nnx.Rngs, 
                 dtype: jnp.dtype=jnp.float32,
                 hidden_dims: list[int]=[32, 64, 128, 256, 512],
                 latent_dim: int=256,
                 gamma: float=1000,
                 max_capacity: int = 50, 
                 capacity_warmup: int = 1e5,
                 ):
        self.rngs = rngs
        self.dtype = dtype
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.max_capacity = max_capacity
        self.capacity_warmup = capacity_warmup

        assert height == width, "Height and width must be equal"
        self.height = height

        # Encoder
        encoder_layers = [
            [
                nnx.Conv(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                rngs=rngs,
                dtype=dtype,
                ),
                nnx.BatchNorm(
                    num_features=out_ch,
                    dtype=dtype,
                    rngs=rngs,
                ),
                nnx.relu,
            ]
            for in_ch, out_ch in zip([channels] + hidden_dims[:-1], hidden_dims)
        ] # dimensions will be min(height, width) / 2^len(hidden_dims)
        self.encoder = nnx.Sequential(*[layer for block in encoder_layers for layer in block])
        self.latent_mean = nnx.Linear(
            in_features=(height // 2**len(hidden_dims))**2 * hidden_dims[-1],
            out_features=latent_dim,
            dtype=dtype,
            rngs=rngs,
        )
        self.latent_log_var = nnx.Linear(
            in_features=(height // 2**len(hidden_dims))**2 * hidden_dims[-1],
            out_features=latent_dim,
            dtype=dtype,
            rngs=rngs,
        )

        # Decoder
        self.decoder_input = nnx.Linear(
            in_features=latent_dim,
            out_features=(height // 2**len(hidden_dims))**2 * hidden_dims[-1],
            dtype=dtype,
            rngs=rngs,
        )
        decoder_layers = [
            [
                nnx.ConvTranspose(
                    in_features=in_ch,
                    out_features=out_ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    rngs=rngs,
                    dtype=dtype,
                ),
                nnx.BatchNorm(
                    num_features=out_ch,
                    dtype=dtype,
                    rngs=rngs,
                ),
                nnx.relu,
            ]
            for in_ch, out_ch in zip(hidden_dims[::-1], hidden_dims[::-1][1:] + [channels])
        ]   
        self.decoder = nnx.Sequential(*[layer for block in decoder_layers for layer in block])

    def reparameterize(self, mean, log_var):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(self.rngs.reparam(), std.shape)
        return mean + std * eps
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.latent_mean(x), self.latent_log_var(x)

    def decode(self, z):
        z = self.decoder_input(z)
        z = rearrange(z, 'b (h w c) -> b h w c', h=self.height // 2**len(self.hidden_dims), w=self.height // 2**len(self.hidden_dims))
        return self.decoder(z)
    
    def __call__(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var
    
    def loss(self, x, step: int):
        '''
        References
        ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        '''
        x_hat, mean, log_var = self(x)
        reconstruction_loss = optax.l2_loss(x, x_hat).mean()
        kl_loss = jnp.sum(0.5 * (mean**2 + jnp.exp(log_var) - 1 - log_var), axis=-1).mean()

        # VAE loss with capacity control
        capacity = jnp.minimum(self.max_capacity, self.max_capacity * step / self.capacity_warmup)
        loss = reconstruction_loss + self.gamma * jnp.abs(kl_loss - capacity)
        return loss, {
            'reconstruction_loss': reconstruction_loss, 
            'kl_loss': kl_loss, 
            'capacity': capacity,
            'predictions': x_hat,
            }

def apply_transforms(transforms, x):
    for transform in transforms:
        x = transform(x)
    return x

def collate_fn(x):
    return np.stack(x)

@nnx.jit
def train_step(model: VAE, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch, step):
    def loss_fn(model, batch, step):
        return model.loss(batch, step)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch, step)
    aux.pop('predictions')
    metrics.update(loss=loss.astype(jnp.float32), **aux)
    optimizer.update(grads=grads)

@nnx.jit
def val_step(model: VAE, batch, step):
    loss, aux = model.loss(batch, step)
    return loss, aux

    
@dataclass
class Args:
    LR: float = 1e-4
    MOMENTUM: float = 0.9
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 8
    SEED: int = 42
    PRECISION: jnp.dtype = jnp.float32
    LOG_INTERVAL: int = 250
    EPOCHS: int = 10

if __name__ == "__main__":
    args = tyro.cli(Args)
    wandb.init(
        project="back2basics",
        name="capacity-control-vae",
        config={
            "lr": args.LR,
            "momentum": args.MOMENTUM,
            "batch_size": args.BATCH_SIZE,
            "num_workers": args.NUM_WORKERS,
            "seed": args.SEED
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
        batch_size=args.BATCH_SIZE, 
        shuffle=True,
        num_workers=args.NUM_WORKERS,
        collate_fn=collate_fn,
        multiprocessing_context=multiprocessing.get_context('spawn') if args.NUM_WORKERS > 0 else None,
        # persistent_workers=NUM_WORKERS > 0,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=args.NUM_WORKERS,
        collate_fn=collate_fn,
        multiprocessing_context=multiprocessing.get_context('spawn') if args.NUM_WORKERS > 0 else None,
    )

    rngs = nnx.Rngs(args.SEED, params=1, dropout=2, reparam=3)

    model = VAE(
        height=224,
        width=224,
        channels=3,
        rngs=rngs,
        dtype=args.PRECISION,
        latent_dim=1024,
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(args.LR, args.MOMENTUM))
    train_metrics = nnx.MultiMetric(
        loss = nnx.metrics.Average("loss"),
        reconstruction_loss = nnx.metrics.Average("reconstruction_loss"),
        kl_loss = nnx.metrics.Average("kl_loss"),
        capacity = nnx.metrics.Average("capacity"),
    )
    
    model.train()
    step = 0
    for epoch in range(args.EPOCHS):
        for batch_num, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch = jnp.array(batch).astype(args.PRECISION)
            train_step(model, optimizer, train_metrics, batch, step)
            if step % args.LOG_INTERVAL == 0:
                all_metrics = train_metrics.compute()
                train_metrics.reset()

                val_losses = []
                model.eval()
                for i, val_batch in enumerate(val_loader):
                    val_batch = jnp.array(val_batch).astype(args.PRECISION)
                    val_loss, aux = val_step(model, val_batch, step)
                    val_losses.append(val_loss)
                    if i == 0:
                        inputs = jax.device_get(val_batch)
                        outputs = jax.device_get(aux['predictions'])

                        sampled_indices = np.random.choice(outputs.shape[0], size=3, replace=False)
                        sampled_inputs = inputs[sampled_indices]
                        sampled_outputs = outputs[sampled_indices]

                        unnormalized_inputs = UnNormalizeTransform.from_stats(mean, std, sampled_inputs)
                        unnormalized_inputs = np.concatenate(unnormalized_inputs, axis=1)
                        unnormalized_outputs = UnNormalizeTransform.from_stats(mean, std, sampled_outputs)
                        unnormalized_outputs = np.concatenate(unnormalized_outputs, axis=1)

                        examples = np.concatenate([unnormalized_inputs, unnormalized_outputs], axis=0).astype(np.float32)
                        all_metrics["examples"] = wandb.Image(examples, caption="Input and Output")
                
                val_loss = np.mean(val_losses)
                all_metrics["val_loss"] = val_loss
                model.train()
                wandb.log(all_metrics, step=step)
            step += 1
