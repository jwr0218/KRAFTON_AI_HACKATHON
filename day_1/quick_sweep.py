"""빠른 스윕: 유망한 config만, 200 에폭, 조기종료"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random, time, sys

class ConceptMultiplier(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        self.emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(24, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, batch_first=True, norm_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        self.head.weight = self.emb.weight

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        return self.head(self.ln_f(self.transformer(h, mask=mask, is_causal=True)))

class MulDS(Dataset):
    def __init__(self, n=100000, seed=0):
        random.seed(seed)
        self.data = []
        for _ in range(n):
            a, b = random.randint(0,63), random.randint(0,63)
            p = a*b
            row = ([int(c) for c in format(a,'06b')[::-1]] +
                   [int(c) for c in format(b,'06b')[::-1]] +
                   [int(c) for c in format(p,'012b')[::-1]])
            self.data.append(torch.tensor(row, dtype=torch.long))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def run(cfg, device, tr, te, max_epochs=200):
    d_model, n_heads, n_layers, d_ff = cfg
    torch.manual_seed(42)
    model = ConceptMultiplier(d_model, n_heads, n_layers, d_ff).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        for batch in tr:
            batch = batch.to(device)
            x, y = batch[:,:-1], batch[:,1:]
            opt.zero_grad()
            loss = crit(model(x)[:,11:].reshape(-1,2), y[:,11:].reshape(-1))
            loss.backward(); opt.step()
        sch.step()

        if (ep+1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in te:
                    batch = batch.to(device)
                    x, y = batch[:,:-1], batch[:,1:]
                    preds = model(x)[:,11:].argmax(-1)
                    correct += (preds == y[:,11:]).all(1).sum().item()
            acc = correct / 10000
            best = max(best, acc)
            if acc >= 0.99:
                print(f"  d={d_model} h={n_heads} l={n_layers} ff={d_ff} | params={params} | acc={acc:.4f} @ ep{ep+1}  ✓ ({time.time()-t0:.0f}s)", flush=True)
                return params, best, True

    print(f"  d={d_model} h={n_heads} l={n_layers} ff={d_ff} | params={params} | best={best:.4f} ✗ ({time.time()-t0:.0f}s)", flush=True)
    return params, best, False

def main():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr = DataLoader(MulDS(100000, 0), batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    te = DataLoader(MulDS(10000,  1), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # 파라미터 오름차순: 각 사이즈에서 가장 가능성 높은 조합
    configs = [
        (8,  2, 2, 16),   # 1426
        (8,  2, 2, 32),   # 1970
        (8,  2, 3, 32),   # 2842
        (16, 2, 2, 32),   # 4898
        (16, 2, 2, 64),   # 7010
        (16, 4, 2, 64),   # 7010
        (16, 2, 3, 64),   # 10290
        (16, 4, 3, 64),   # 10290
        (24, 4, 2, 64),   # 11986
        (24, 4, 2, 96),   # 15122
        (32, 4, 2, 128),  # 26306
        (32, 4, 3, 128),  # 39010 (원본)
    ]

    print("\n=== 스윕 시작 (작은→큰 순, 99% 달성 시 표시) ===\n")
    found = None
    for cfg in configs:
        params, best, ok = run(cfg, device, tr, te)
        if ok and found is None:
            found = (cfg, params)

    print("\n" + "="*60)
    if found:
        cfg, p = found
        d,h,l,f = cfg
        print(f"최소 config: d_model={d}, n_heads={h}, n_layers={l}, d_ff={f}")
        print(f"Params: {p}  (원본 39010 → {p/39010*100:.1f}%)")
    else:
        print("99% 달성 config 없음")

if __name__ == '__main__':
    main()
