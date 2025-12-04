import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ======================
# Config
# ======================

SEED = 0
BATCH = 128
LR_RBM = 0.05
AE_SIZES = (784, 1000, 500, 250, 30)
STEPS_RBM1 = 800
STEPS_RBM2 = 800
STEPS_RBM3 = 800
STEPS_RBM4 = 800

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.manual_seed(SEED)


# ======================
# Dataset (MNIST, flattened)
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
# RBM definition
# ======================

class BernoulliRBM(nn.Module):

    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid

        # W: [n_vis, n_hid], bv: [n_vis], bh: [n_hid]
        self.W = nn.Parameter(0.01 * torch.randn(n_vis, n_hid))
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))

    def p_h_given_v(self, v):
        # v: [B, n_vis] -> p(h=1|v): [B, n_hid]
        return torch.sigmoid(v @ self.W + self.bh)

    def p_v_given_h(self, h):
        # h: [B, n_hid] -> p(v=1|h): [B, n_vis]
        return torch.sigmoid(h @ self.W.t() + self.bv)

    def cd1_step(self, v0):
        # ---- Positive phase ----
        ph0 = self.p_h_given_v(v0)              # [B, n_hid]
        h0 = torch.bernoulli(ph0)               # sample

        # ---- Negative phase ----
        pv1 = self.p_v_given_h(h0)              # [B, n_vis]
        v1 = torch.bernoulli(pv1)               # sample
        ph1 = self.p_h_given_v(v1)              # [B, n_hid]

        # Statistics
        pos = v0.t() @ ph0    # [n_vis, n_hid]
        neg = v1.t() @ ph1    # [n_vis, n_hid]

        return pos, neg, v0, pv1, ph0, ph1

class GaussianTopRBM(nn.Module):

    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.W = nn.Parameter(0.01 * torch.randn(n_vis, n_hid))
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))

    # mean của hidden Gaussian, dùng như "p(h|v)"
    def p_h_given_v(self, v):
        # Không sigmoid, vì hidden là Gaussian linear
        return v @ self.W + self.bh       # [B, n_hid]

    def sample_h(self, v):
        mean = self.p_h_given_v(v)
        h = mean + torch.randn_like(mean)  # N(mean, 1)
        return h, mean

    def p_v_given_h(self, h):
        # visible logistic
        return torch.sigmoid(h @ self.W.t() + self.bv)

    def sample_v(self, h):
        pv = self.p_v_given_h(h)
        v = torch.bernoulli(pv)
        return v, pv

    def cd1_step(self, v0):
        # ----- Positive phase -----
        h0, mean0 = self.sample_h(v0)

        # ----- Negative phase -----
        v1, pv1 = self.sample_v(h0)
        h1, mean1 = self.sample_h(v1)

        # Dùng mean (kỳ vọng) để tính pos/neg
        pos = v0.t() @ mean0
        neg = v1.t() @ mean1

        # Trả về mean0/mean1 như ph0/ph1 cho train_rbm()
        return pos, neg, v0, pv1, mean0, mean1


# ======================
# Training RBM
# ======================

def train_rbm(rbm, loader, steps=800, lr=0.05, name="RBM"):
    rbm.to(DEVICE)
    opt = optim.SGD(rbm.parameters(), lr=lr)

    it = 0
    print(f"Training {name} ...")
    for x, _ in loader:
        v0 = x.to(DEVICE, non_blocking=True)

        # CD-1
        with torch.no_grad():
            pos, neg, v0, pv1, ph0, ph1 = rbm.cd1_step(v0)

        B = v0.size(0)
        opt.zero_grad(set_to_none=True)

        # Gradients
        rbm.W.grad  = - (pos - neg) / B
        rbm.bv.grad = - (v0 - pv1).mean(dim=0)
        rbm.bh.grad = - (ph0 - ph1).mean(dim=0)

        opt.step()

        if it % 50 == 0:
            with torch.no_grad():
                recon_bce = F.binary_cross_entropy(pv1, v0, reduction="mean").item()
            print(f"[{name}] step {it:4d}/{steps} | recon_BCE = {recon_bce:.4f}")

        it += 1
        if it >= steps:
            break

    print(f"Finished {name}.")
    return rbm


# ======================
# Transform data with RBM (for stacking)
# ======================

def transform_with_rbm(rbm, loader):
    rbm.eval()
    H = []

    with torch.no_grad():
        for x, _ in loader:
            v = x.to(DEVICE)
            ph = rbm.p_h_given_v(v)   # [B, n_hidden]
            H.append(ph.cpu())

    H = torch.cat(H, dim=0)
    print("Hidden feature shape:", H.shape)
    return H


def make_loader_from_tensor(tensor, batch_size=BATCH):
    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(tensor, torch.zeros(len(tensor)))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return loader


def main():
    # ---------- RBM1: 784 -> 1000 ----------
    rbm1 = BernoulliRBM(784, 1000)
    rbm1 = train_rbm(rbm1, train_loader, steps=STEPS_RBM1, lr=LR_RBM, name="RBM1 784->1000")

    # Features for RBM2
    H1 = transform_with_rbm(rbm1, train_loader)         # [N,1000]
    loader2 = make_loader_from_tensor(H1)

    # ---------- RBM2: 1000 -> 500 ----------
    rbm2 = BernoulliRBM(1000, 500)
    rbm2 = train_rbm(rbm2, loader2, steps=STEPS_RBM2, lr=LR_RBM, name="RBM2 1000->500")

    # Features for RBM3
    H2 = transform_with_rbm(rbm2, loader2)             # [N,500]
    loader3 = make_loader_from_tensor(H2)

    # ---------- RBM3: 500 -> 250 ----------
    rbm3 = BernoulliRBM(500, 250)
    rbm3 = train_rbm(rbm3, loader3, steps=STEPS_RBM3, lr=LR_RBM, name="RBM3 500->250")

    # Features for RBM4
    H3 = transform_with_rbm(rbm3, loader3)             # [N,250]
    loader4 = make_loader_from_tensor(H3)

    # ---------- RBM4: 250 -> 30 ----------
    rbm4 = BernoulliRBM(250, 30)
    rbm4 = train_rbm(rbm4, loader4, steps=STEPS_RBM4, lr=LR_RBM, name="Top RBM (Gaussian) 250->30")

    # Save RBMs
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(rbm1.state_dict(), "checkpoints/rbm1_784_1000.pth")
    torch.save(rbm2.state_dict(), "checkpoints/rbm2_1000_500.pth")
    torch.save(rbm3.state_dict(), "checkpoints/rbm3_500_250.pth")
    torch.save(rbm4.state_dict(), "checkpoints/rbm4_250_30.pth")

    print("Saved RBM weights to 'checkpoints/' directory.")


if __name__ == "__main__":
    main()
