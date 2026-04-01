"""
1_2.py 스타일 ConceptMultiplier 최소 파라미터 탐색.
작은 config부터 순서대로 학습, 99% 달성 여부 확인.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import time

class ConceptMultiplier(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        self.emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(24, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        self.head.weight = self.emb.weight

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x_emb = self.emb(x) + self.pos_emb(positions).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        return self.head(self.ln_f(out))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MultiplierDataset(Dataset):
    def __init__(self, num_samples=100000, seed=None):
        if seed is not None:
            random.seed(seed)
        self.data = []
        for _ in range(num_samples):
            a, b = random.randint(0, 63), random.randint(0, 63)
            p = a * b
            a_bin = [int(x) for x in format(a, '06b')[::-1]]
            b_bin = [int(x) for x in format(b, '06b')[::-1]]
            p_bin = [int(x) for x in format(p, '012b')[::-1]]
            self.data.append(torch.tensor(a_bin + b_bin + p_bin, dtype=torch.long))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def run_config(d_model, n_heads, n_layers, d_ff, device, train_loader, test_loader, epochs=200):
    torch.manual_seed(42)
    model = ConceptMultiplier(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff).to(device)
    params = count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    achieved_99 = False

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 11:, :].reshape(-1, 2), y[:, 11:].reshape(-1))
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    x, y = batch[:, :-1], batch[:, 1:]
                    logits = model(x)
                    preds = logits[:, 11:, :].argmax(dim=-1)
                    correct += (preds == y[:, 11:]).all(dim=1).sum().item()
            acc = correct / 10000
            if acc > best_acc:
                best_acc = acc
            if acc >= 0.99 and not achieved_99:
                achieved_99 = True

    return params, best_acc, achieved_99

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("데이터셋 생성 중 (공유)...")
    train_loader = DataLoader(MultiplierDataset(100000, seed=0), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(MultiplierDataset(10000, seed=1), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # 작은 것부터 순서대로 (params 오름차순)
    configs = [
        (8,  2, 2, 16),
        (8,  2, 2, 32),
        (8,  2, 3, 32),
        (16, 2, 2, 32),
        (16, 2, 2, 64),
        (16, 4, 2, 64),
        (16, 2, 3, 64),
        (16, 4, 3, 64),
        (24, 4, 2, 64),
        (24, 4, 2, 96),
        (24, 4, 3, 64),
        (32, 4, 2, 128),
        (32, 4, 3, 128),  # 원본
    ]

    print("\n%8s %8s %9s %6s %8s %8s %10s" % (
        "d_model","n_heads","n_layers","d_ff","params","best_acc",">=99%"))
    print("-" * 65)

    found_min = None
    for d_model, n_heads, n_layers, d_ff in configs:
        t0 = time.time()
        params, best_acc, ok = run_config(
            d_model, n_heads, n_layers, d_ff,
            device, train_loader, test_loader, epochs=200
        )
        elapsed = time.time() - t0
        flag = "YES" if ok else "no"
        print("%8d %8d %9d %6d %8d %8.4f %10s  (%.0fs)" % (
            d_model, n_heads, n_layers, d_ff, params, best_acc, flag, elapsed))

        if ok and found_min is None:
            found_min = (d_model, n_heads, n_layers, d_ff, params)

    print("\n" + "=" * 65)
    if found_min:
        d, h, l, f, p = found_min
        print(f"최소 99% 달성 config: d_model={d}, n_heads={h}, n_layers={l}, d_ff={f}")
        print(f"Params: {p}  (원본 39010에서 {39010-p:+d}, {p/39010*100:.1f}%)")
    else:
        print("99% 달성 config 없음 — epochs 늘리거나 다른 config 필요")

if __name__ == '__main__':
    main()
