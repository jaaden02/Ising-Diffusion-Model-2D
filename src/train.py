import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import IsingDataset
from src.model import UNet_S
from src.diffusion import Diffuser

class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: p.clone().detach()
            for name, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].sub_((1 - self.decay) * (self.shadow[name] - p.data))

    def apply(self, model):
        self.backup = {
            name: p.clone() for name, p in model.named_parameters() if p.requires_grad
        }
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.backup[name])


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False, True
        self.counter += 1
        return self.counter >= self.patience, False


def save_checkpoint(path, model, optimizer, ema, epoch, loss_history):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "epoch": epoch,
        "loss_history": loss_history
    }, path)


def load_checkpoint(path, model, optimizer, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and ckpt["ema"] is not None:
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
    return ckpt.get("epoch", 0), ckpt.get("loss_history", {})


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {args.data}")
    full_dataset = IsingDataset(args.data, augment=True)
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Dataset Size: {len(full_dataset)} total (Train: {train_size}, Val: {val_size})")
    print(f"Grid Dimensions: L={full_dataset.L}")

    model = UNet_S().to(device)
    diffuser = Diffuser(schedule=args.schedule).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / 200)
    )
    ema = EMA(model) if args.use_ema else None
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=args.patience) if args.patience else None

    start_epoch = 0
    loss_history = {"train_mse": [], "train_vlb": [], "val_mse": [], "val_vlb": []}
    if args.resume:
        start_epoch, loss_history = load_checkpoint(args.resume, model, optimizer, ema, device)
        print(f"Resumed training from epoch {start_epoch}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Running options: EMA={args.use_ema}, VLB={args.use_vlb}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_train_mse = 0.0
        epoch_train_vlb = 0.0
        n_train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for configs, T_phys in pbar:
            configs = configs.to(device)
            T_phys = T_phys.to(device)

            t = torch.randint(0, diffuser.diff_steps, (configs.shape[0],), device=device)
            x_t, eps = diffuser.diffuse(configs, t)

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                model_output = model(x_t, t.float(), T_phys)
                eps_pred, v_pred = model_output.chunk(2, dim=1)

                loss_mse = F.mse_loss(eps_pred, eps)
                if args.use_vlb:
                    loss_vlb = diffuser.vlb_loss(configs, x_t, t, eps_pred.detach(), v_pred)
                    loss = loss_mse + args.lambda_vlb * loss_vlb
                else:
                    loss_vlb = torch.tensor(0.0, device=device)
                    loss = loss_mse

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            if ema is not None:
                ema.update(model)

            epoch_train_mse += loss_mse.item()
            epoch_train_vlb += loss_vlb.item()
            n_train_batches += 1

            pbar.set_postfix({
                "mse": f"{loss_mse.item():.4f}",
                "vlb": f"{loss_vlb.item():.4f}" if args.use_vlb else "N/A"
            })

        avg_train_mse = epoch_train_mse / n_train_batches
        avg_train_vlb = epoch_train_vlb / n_train_batches
        loss_history["train_mse"].append(avg_train_mse)
        loss_history["train_vlb"].append(avg_train_vlb)

        model.eval()
        if ema is not None:
            ema.apply(model)

        epoch_val_mse = 0.0
        epoch_val_vlb = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for configs, T_phys in val_loader:
                configs = configs.to(device)
                T_phys = T_phys.to(device)
                t = torch.randint(0, diffuser.diff_steps, (configs.shape[0],), device=device)
                x_t, eps = diffuser.diffuse(configs, t)

                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    model_output = model(x_t, t.float(), T_phys)
                    eps_pred, v_pred = model_output.chunk(2, dim=1)

                    loss_mse = F.mse_loss(eps_pred, eps)
                    if args.use_vlb:
                        loss_vlb = diffuser.vlb_loss(configs, x_t, t, eps_pred.detach(), v_pred)
                    else:
                        loss_vlb = torch.tensor(0.0, device=device)

                epoch_val_mse += loss_mse.item()
                epoch_val_vlb += loss_vlb.item()
                n_val_batches += 1

        if ema is not None:
            ema.restore(model)

        avg_val_mse = epoch_val_mse / n_val_batches
        avg_val_vlb = epoch_val_vlb / n_val_batches
        loss_history["val_mse"].append(avg_val_mse)
        loss_history["val_vlb"].append(avg_val_vlb)

        print(f"Epoch {epoch + 1}: Train MSE = {avg_train_mse:.5f} | Val MSE = {avg_val_mse:.5f}")

        if (epoch + 1) % args.ckpt_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, ema, epoch + 1, loss_history)
            print(f"Saved periodic checkpoint: {ckpt_path}")

        if early_stopping is not None:
            stop_training, is_best = early_stopping.step(avg_val_mse)
            if is_best:
                best_path = ckpt_dir / "ckpt_best.pt"
                save_checkpoint(best_path, model, optimizer, ema, epoch + 1, loss_history)
                print(f"New best validation loss! Saved checkpoint to {best_path}")
            if stop_training:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    with open(ckpt_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)
    print("Training finished! Saved training loss history.")

def main():
    parser = argparse.ArgumentParser(description="Train temperature-conditioned 2D Ising Diffusion Model")
    parser.add_argument("--data", type=str, required=True, help="Path to Ising dataset (.npz)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set fraction")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Maintain EMA weights")
    parser.add_argument("--use-vlb", action="store_true", help="Add VLB loss term to predict variance")
    parser.add_argument("--lambda-vlb", type=float, default=0.001, help="Weight scale factor for VLB term")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine"], help="Noise schedule name")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader subprocesses worker count")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory where weights checkpoints will be stored")
    parser.add_argument("--ckpt-every", type=int, default=10, help="Epoch frequency for periodic checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint file")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
