# rbm_pretrain_ae.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------ Config ------------------------
SEED = 0
BATCH = 128
LR_RBM = 0.05       
STEPS_RBM1 = 800    
STEPS_RBM2 = 800    
EPOCHS_AE = 5       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)

# ------------------------ Data --------------------------
def flatten(x): return x.view(-1)

tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flatten)])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=(DEVICE.type=="cuda"))
print(f"Loaded MNIST: {len(train_ds)} train — device={DEVICE}")

# ------------------------ RBM ---------------------------
class BernoulliRBM(nn.Module):
    """
    Bernoulli-Bernoulli RBM.
    Visible units are treated as probabilities in [0,1] (no hard binarization needed).
    """
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.W  = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.bv = nn.Parameter(torch.zeros(n_vis))
        self.bh = nn.Parameter(torch.zeros(n_hid))

    def p_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.bh)

    def p_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.bv)

    @torch.no_grad()
    def cd1_step(self, v0):
        ph0 = self.p_h_given_v(v0)
        h0  = torch.bernoulli(ph0)

        pv1 = self.p_v_given_h(h0)
        v1  = torch.bernoulli(pv1)

        ph1 = self.p_h_given_v(v1)

        pos = v0.t() @ ph0
        neg = v1.t() @ ph1
        return pos, neg, v0, pv1, ph0, ph1

def train_rbm(rbm, loader, steps=800, lr=0.05, name="RBM"):
    rbm.to(DEVICE)
    opt = torch.optim.SGD(rbm.parameters(), lr=lr)
    it = 0
    print(f"Training {name} with CD-1 for ~{steps} steps...")
    for x, _ in loader:
        v0 = x.to(DEVICE, non_blocking=True)  
        pos, neg, v0, pv1, ph0, ph1 = rbm.cd1_step(v0)

        opt.zero_grad(set_to_none=True)
        B = v0.size(0)
        rbm.W.grad  = - (pos - neg) / B
        rbm.bv.grad = - (v0 - pv1).mean(dim=0)
        rbm.bh.grad = - (ph0 - ph1).mean(dim=0)
        opt.step()

        if it % 50 == 0:
            with torch.no_grad():
                recon_bce = F.binary_cross_entropy(pv1, v0, reduction="mean").item()
            print(f"[{name} step {it:4d}] recon_BCE={recon_bce:.4f}")
        it += 1
        if it >= steps:
            break
    print(f"Finished {name}.")
    return rbm

@torch.no_grad()
def transform_with_rbm(rbm, loader):
    """Pass data through RBM's hidden mean activations to build features for next RBM."""
    H = []
    for x, _ in loader:
        v = x.to(DEVICE)
        ph = rbm.p_h_given_v(v)  
        H.append(ph.cpu())
    return torch.cat(H, dim=0)  

class AE(nn.Module):
    def __init__(self, sizes=(784,256,64)):
        super().__init__()
        d, h1, h2 = sizes
        self.enc1 = nn.Linear(d,  h1)
        self.enc2 = nn.Linear(h1, h2)
        self.dec2 = nn.Linear(h2, h1)
        self.dec1 = nn.Linear(h1, d)
        self.act  = nn.ReLU()
        self.out  = nn.Sigmoid()

    def forward(self, x):
        z1 = self.act(self.enc1(x))
        z2 = self.enc2(z1)           
        y1 = self.act(self.dec2(z2))
        y  = self.out(self.dec1(y1))
        return y, z2

def initialize_ae_from_rbms(ae, rbm1, rbm2):
    ae.enc1.weight.data = rbm1.W.t().detach().clone()
    ae.enc1.bias.data   = rbm1.bh.detach().clone()
    ae.enc2.weight.data = rbm2.W.t().detach().clone()
    ae.enc2.bias.data   = rbm2.bh.detach().clone()
    ae.dec2.weight.data = rbm2.W.detach().clone()
    ae.dec2.bias.data   = rbm2.bv.detach().clone()
    ae.dec1.weight.data = rbm1.W.detach().clone()
    ae.dec1.bias.data   = rbm1.bv.detach().clone()

# ------------------------ Pipeline ----------------------
def main():
    rbm1 = BernoulliRBM(784, 256)
    rbm1 = train_rbm(rbm1, train_loader, steps=STEPS_RBM1, lr=LR_RBM, name="RBM1 784->256")

    print("Building hidden features from RBM1 for RBM2...")
    H = transform_with_rbm(rbm1, train_loader)             
    hid_ds = torch.utils.data.TensorDataset(H, torch.zeros(len(H)))  
    hid_loader = DataLoader(hid_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    rbm2 = BernoulliRBM(256, 64)
    def gen_loader_from_tensor(tensor):
        ds = torch.utils.data.TensorDataset(tensor, torch.zeros(len(tensor)))
        return DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    rbm2 = train_rbm(rbm2, hid_loader, steps=STEPS_RBM2, lr=LR_RBM, name="RBM2 256->64")

    ae = AE((784,256,64)).to(DEVICE)
    initialize_ae_from_rbms(ae, rbm1, rbm2)
    print("Initialized AE from stacked RBMs. Starting fine-tuning...")

    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    ae.train()
    for epoch in range(1, EPOCHS_AE+1):
        total = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            x_hat, _ = ae(x)
            loss = crit(x_hat, x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0)
        print(f"[AE fine-tune] epoch {epoch}/{EPOCHS_AE}  loss={total/len(train_loader.dataset):.4f}")

    ae.eval()
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(DEVICE)
        x_hat, _ = ae(x)
    n = 10
    fig, axes = plt.subplots(2, n, figsize=(n*1.2, 3.0))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().view(28,28), cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].cpu().view(28,28), cmap="gray"); axes[1, i].axis("off")
    fig.suptitle("Top: Original — Bottom: Reconstruction (RBM-pretrained AE)", y=1.02)
    plt.tight_layout()
    plt.savefig("recon_rbm.png", dpi=200)
    plt.show()
    print("Saved recon_rbm.png")

if __name__ == "__main__":
    main()