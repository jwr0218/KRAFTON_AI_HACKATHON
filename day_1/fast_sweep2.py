"""d_model=16부터 유망 config만, 100 에폭"""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random, time

class ConceptMultiplier(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff):
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
    def __init__(self, n, seed):
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

def run_one(cfg, device, tr, te, max_epochs=100):
    d_model, n_heads, n_layers, d_ff = cfg
    torch.manual_seed(42)
    model = ConceptMultiplier(d_model, n_heads, n_layers, d_ff).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt  = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch  = CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        for batch in tr:
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:,:-1], batch[:,1:]
            opt.zero_grad()
            crit(model(x)[:,11:].reshape(-1,2), y[:,11:].reshape(-1)).backward()
            opt.step()
        sch.step()
        if (ep+1) % 10 == 0:
            model.eval(); correct = 0
            with torch.no_grad():
                for batch in te:
                    batch = batch.to(device, non_blocking=True)
                    x, y = batch[:,:-1], batch[:,1:]
                    correct += (model(x)[:,11:].argmax(-1) == y[:,11:]).all(1).sum().item()
            acc = correct / 10000
            best = max(best, acc)
            print(f"  [{d_model},{n_heads},{n_layers},{d_ff}] ep{ep+1}: {acc:.4f}", flush=True)
            if acc >= 0.99:
                return params, best, True, ep+1, time.time()-t0
    return params, best, False, max_epochs, time.time()-t0

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    tr = DataLoader(MulDS(100000, 0), batch_size=512, shuffle=True,  num_workers=4, pin_memory=True)
    te = DataLoader(MulDS(10000,  1), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # d_model=16부터, params 오름차순
    configs = [
        (16, 2, 2, 32),   # 4898
        (16, 4, 2, 64),   # 7010
        (16, 2, 2, 64),   # 7010
        (16, 2, 3, 64),   # 10290
        (16, 4, 3, 64),   # 10290
        (24, 4, 2, 64),   # 11986
        (24, 4, 2, 96),   # 15122
        (32, 4, 2, 128),  # 26306
    ]

    print(f"{'config':28s} {'params':>7} {'best':>7} {'ok':>6} {'ep':>5} {'sec':>6}")
    print("-"*58)
    for cfg in configs:
        params, best, ok, ep, sec = run_one(cfg, device, tr, te, max_epochs=100)
        d,h,l,f = cfg
        tag = "YES ✓" if ok else "no"
        print(f"d{d} h{h} l{l} ff{f:<4} {params:>7}  {best:.4f}  {tag:>6}  {ep:>4}  {sec:>5.0f}s", flush=True)
        if ok:
            print(f"\n→ 최소 99% config: d_model={d}, n_heads={h}, n_layers={l}, d_ff={f}, params={params} ({params/39010*100:.1f}% of 39010)\n")
            break  # 최소 찾으면 종료
