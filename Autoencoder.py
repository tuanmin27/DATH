import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# ======================
# Config (must match pretrain_rbm.py)
# ======================

SEED = 0
BATCH = 128
LR_RBM = 0.05
AE_SIZES = (784, 1000, 500, 250, 30)
EPOCHS_AE = 10
N_CG = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.manual_seed(SEED)


# ======================
# Dataset MNIST (for fine-tuning + visualization)
# ======================

def flatten(x):
    return x.view(-1)   # 28x28 => 784


tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(flatten)
])

train_ds = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=tfm
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print("Train size:", len(train_ds))


# ======================
# RBM definition (same as in pretrain_rbm.py)
#   -> used only to hold weights loaded from checkpoints
# ======================

class BernoulliRBM(nn.Module):

    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.W = nn.Parameter(torch.zeros(n_vis, n_hid))
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))

    def p_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.bh)

    def p_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.bv)


# ======================
# Autoencoder
# ======================

class AE(nn.Module):
    def __init__(self, sizes=(784, 1000, 500, 250, 30)):
        super().__init__()
        d, h1, h2, h3, code = sizes

        # Encoder
        self.enc1 = nn.Linear(d,   h1)
        self.enc2 = nn.Linear(h1,  h2)
        self.enc3 = nn.Linear(h2,  h3)
        self.enc4 = nn.Linear(h3,  code)   # code layer (linear)

        # Decoder (symmetric)
        self.dec4 = nn.Linear(code, h3)
        self.dec3 = nn.Linear(h3,   h2)
        self.dec2 = nn.Linear(h2,   h1)
        self.dec1 = nn.Linear(h1,   d)

        # Activation: logistic, output logistic
        self.act = nn.Sigmoid()
        self.out = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        z1 = self.act(self.enc1(x))   # 1000 logistic
        z2 = self.act(self.enc2(z1))  # 500 logistic
        z3 = self.act(self.enc3(z2))  # 250 logistic
        code = self.enc4(z3)          # 30 linear

        # Decoder
        y3 = self.act(self.dec4(code))
        y2 = self.act(self.dec3(y3))
        y1 = self.act(self.dec2(y2))
        y  = self.out(self.dec1(y1))  # logistic output

        return y, code


# ======================
# Initialize AE from RBMs
# ======================

def initialize_ae_from_rbms(ae, rbms):
    rbm1, rbm2, rbm3, rbm4 = rbms

    with torch.no_grad():
        # encoder 1: 784 -> 1000
        ae.enc1.weight.copy_(rbm1.W.t())
        ae.enc1.bias.copy_(rbm1.bh)

        # encoder 2: 1000 -> 500
        ae.enc2.weight.copy_(rbm2.W.t())
        ae.enc2.bias.copy_(rbm2.bh)

        # encoder 3: 500 -> 250
        ae.enc3.weight.copy_(rbm3.W.t())
        ae.enc3.bias.copy_(rbm3.bh)

        # encoder 4 (code): 250 -> 30
        ae.enc4.weight.copy_(rbm4.W.t())
        ae.enc4.bias.copy_(rbm4.bh)

        # decoder 4: 30 -> 250
        ae.dec4.weight.copy_(rbm4.W)
        ae.dec4.bias.copy_(rbm4.bv)

        # decoder 3: 250 -> 500
        ae.dec3.weight.copy_(rbm3.W)
        ae.dec3.bias.copy_(rbm3.bv)

        # decoder 2: 500 -> 1000
        ae.dec2.weight.copy_(rbm2.W)
        ae.dec2.bias.copy_(rbm2.bv)

        # decoder 1: 1000 -> 784
        ae.dec1.weight.copy_(rbm1.W)
        ae.dec1.bias.copy_(rbm1.bv)

    print("Initialized DeepAE from 4 stacked RBMs.")


# ======================
# Fine-tune with LBFGS (CG-like)
# ======================

def fine_tune_with_cg(model, X, epochs=20):
    model.to(DEVICE)
    model.train()

    criterion = nn.MSELoss()

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50
    )

    loss_history = []

    for epoch in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            y_hat, _ = model(X)
            loss = criterion(y_hat, X)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        loss_history.append(loss.detach())
        print(f"[CG] epoch {epoch}/{epochs} | loss = {loss.item():.6f}")

    return model, loss_history


def main():
    # Build X_cg: first N_CG training images
    X_cg_list = []
    for i in range(N_CG):
        x, _ = train_ds[i]
        X_cg_list.append(x)

    X_cg = torch.stack(X_cg_list, dim=0).to(DEVICE)  # [N_CG, 784]
    print("X_cg shape:", X_cg.shape)

    # Load pretrained RBMs
    ckpt_dir = "checkpoints"
    rbm1 = BernoulliRBM(784, 1000)
    rbm2 = BernoulliRBM(1000, 500)
    rbm3 = BernoulliRBM(500, 250)
    rbm4 = BernoulliRBM(250, 30)

    rbm1.load_state_dict(torch.load(os.path.join(ckpt_dir, "rbm1_784_1000.pth"), map_location=DEVICE))
    rbm2.load_state_dict(torch.load(os.path.join(ckpt_dir, "rbm2_1000_500.pth"), map_location=DEVICE))
    rbm3.load_state_dict(torch.load(os.path.join(ckpt_dir, "rbm3_500_250.pth"), map_location=DEVICE))
    rbm4.load_state_dict(torch.load(os.path.join(ckpt_dir, "rbm4_250_30.pth"), map_location=DEVICE))

    rbms = [rbm1.to(DEVICE), rbm2.to(DEVICE), rbm3.to(DEVICE), rbm4.to(DEVICE)]

    # Build AE, initialize from RBMs
    deep_ae = AE(AE_SIZES).to(DEVICE)
    initialize_ae_from_rbms(deep_ae, rbms)

    # Fine-tune
    deep_ae, loss_history = fine_tune_with_cg(deep_ae, X_cg, epochs=EPOCHS_AE)

    # ---- Plot a few reconstructions ----
    deep_ae.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))[0].to(DEVICE)
        recon, _ = deep_ae(batch)

    batch = batch.cpu().view(-1, 28, 28)
    recon = recon.cpu().view(-1, 28, 28)

    n = 10
    plt.figure(figsize=(2*n, 4))

    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(batch[i], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        # Reconstructed
        plt.subplot(2, n, n + i + 1)
        plt.imshow(recon[i], cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()

    # ---- Plot loss ----
    plt.figure(figsize=(6, 4))
    plot_losses = [loss.cpu().item() for loss in loss_history]
    plt.plot(range(1, len(plot_losses) + 1), plot_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (BCE)")
    plt.title("AE Fine-tuning Loss per Epoch")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
