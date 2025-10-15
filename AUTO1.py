
"""
AUTO.py - Entry script: RBM pretraining -> init AE -> fine-tune
Bổ sung:
- Lưu checkpoint mỗi epoch: outputs/checkpoints/ae_epochXXX.pt
- Ghi log loss theo epoch:  outputs/loss_log.csv
- Lưu hình tái tạo:        outputs/recon_rbm.png
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Import pipeline components từ file RBM
from rbm_autoencoder import (
    BernoulliRBM, train_rbm, transform_with_rbm,
    AE, initialize_ae_from_rbms,
    DEVICE, train_loader
)

# ------------------- Config -------------------
EPOCHS_AE = 5
LR_FT     = 1e-3
CKPT_DIR  = "outputs/checkpoints"
LOG_CSV   = "outputs/loss_log.csv"
IMG_PATH  = "outputs/recon_rbm.png"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

def fine_tune_with_logging(ae, train_loader, epochs=5, lr=1e-3, log_csv=LOG_CSV, ckpt_dir=CKPT_DIR):
    crit = nn.BCELoss()
    opt  = optim.Adam(ae.parameters(), lr=lr)

    # Prepare CSV
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss"])

    for epoch in range(1, epochs+1):
        ae.train()
        total = n = 0
        for x, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            x_hat, _ = ae(x)
            loss = crit(x_hat, x)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = x.size(0); total += loss.item() * bs; n += bs
        train_loss = total / n
        print(f"[AE fine-tune] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}")

        # Save checkpoint per epoch
        ckpt_path = os.path.join(ckpt_dir, f"ae_epoch{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": ae.state_dict(),
            "optimizer_state": opt.state_dict(),
            "config": {"epochs": epochs, "lr": lr}
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # Append log
        with open(log_csv, "a", newline="") as f:
            w = csv.writer(f); w.writerow([epoch, train_loss])

def save_recon_grid(ae, loader, out_path=IMG_PATH, n=10):
    ae.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x = x.to(DEVICE)
        x_hat, _ = ae(x)
    fig, axes = plt.subplots(2, n, figsize=(n*1.2, 3.0))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().view(28,28), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].cpu().view(28,28), cmap="gray"); axes[1, i].axis("off")
    fig.suptitle("Top: Original — Bottom: Reconstruction (RBM-pretrained AE)", y=1.02)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close(fig)
    print("Saved image:", out_path)

def main():
    # 1) Train RBM1 (784->256)
    rbm1 = BernoulliRBM(784, 256)
    rbm1 = train_rbm(rbm1, train_loader, steps=800, lr=0.05, name="RBM1 784->256")

    # 2) Features cho RBM2
    print("Building hidden features from RBM1 for RBM2...")
    H = transform_with_rbm(rbm1, train_loader)
    hid_ds = TensorDataset(H, torch.zeros(len(H)))
    hid_loader = DataLoader(hid_ds, batch_size=128, shuffle=True, num_workers=0)

    # 3) Train RBM2 (256->64)
    rbm2 = BernoulliRBM(256, 64)
    rbm2 = train_rbm(rbm2, hid_loader, steps=800, lr=0.05, name="RBM2 256->64")

    # 4) Init AE từ RBMs
    ae = AE((784,256,64)).to(DEVICE)
    initialize_ae_from_rbms(ae, rbm1, rbm2)
    print("Initialized AE from stacked RBMs. Starting fine-tuning...")

    # 5) Fine-tune + log + checkpoint
    fine_tune_with_logging(ae, train_loader, epochs=EPOCHS_AE, lr=LR_FT)

    # 6) Lưu ảnh tái tạo
    save_recon_grid(ae, train_loader, out_path=IMG_PATH)

if __name__ == "__main__":
    main()
