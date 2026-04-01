"""
1_2 학습 실행 + 보고서용 데이터 수집 (99% 달성 시 종료)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import time
import json

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

def train_config(d_model, n_heads, n_layers, d_ff, device, train_loader, test_loader,
                 max_epochs=200, stop_at_99=True, verbose=True):
    torch.manual_seed(42)
    model = ConceptMultiplier(d_model, n_heads, n_layers, d_ff).to(device)
    params = count_parameters(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    history = []  # (epoch, loss, acc)
    best_acc = 0.0
    achieved_epoch = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[:, 11:, :].reshape(-1, 2), y[:, 11:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # 매 에폭 평가
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
        best_acc = max(best_acc, acc)
        history.append((epoch + 1, avg_loss, acc))

        if verbose and ((epoch + 1) % 10 == 0 or acc >= 0.99):
            print(f"  Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        if acc >= 0.99 and achieved_epoch is None:
            achieved_epoch = epoch + 1
            print(f"  *** 99% 달성! (epoch {epoch+1}, params={params}) ***")
            if stop_at_99:
                break

    return params, best_acc, achieved_epoch, history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("데이터셋 생성 중...")
    train_loader = DataLoader(MultiplierDataset(100000, seed=0), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(MultiplierDataset(10000,  seed=1), batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    # ── 메인 모델 학습 곡선 (99% 달성 시 조기 종료) ──────────────────────
    print("\n[MAIN] ConceptMultiplier(32,4,3,128) 학습 시작")
    params, best_acc, ep99, history = train_config(
        32, 4, 3, 128, device, train_loader, test_loader,
        max_epochs=200, stop_at_99=True, verbose=True
    )
    print(f"[MAIN] 완료: params={params}, best_acc={best_acc*100:.2f}%, 99%달성에폭={ep99}")

    # ── 모델 크기 스윕 (각 config 50 에폭, 빠른 비교) ────────────────────
    print("\n[SWEEP] 모델 크기별 정확도 비교 (각 50 에폭)")
    sweep_configs = [
        (8,  2, 2, 16,  "tiny"),
        (8,  2, 2, 32,  "small-a"),
        (16, 2, 2, 64,  "small-b"),
        (16, 4, 3, 64,  "medium"),
        (24, 4, 3, 96,  "medium-L"),
        (32, 4, 3, 128, "original"),
    ]

    sweep_results = []
    for d_model, n_heads, n_layers, d_ff, label in sweep_configs:
        print(f"\n  [{label}] d={d_model}, h={n_heads}, L={n_layers}, ff={d_ff}")
        p, acc50, _, _ = train_config(
            d_model, n_heads, n_layers, d_ff,
            device, train_loader, test_loader,
            max_epochs=50, stop_at_99=False, verbose=False
        )
        print(f"  params={p:,}, acc@50ep={acc50*100:.2f}%")
        sweep_results.append({"label": label, "params": p, "acc": round(acc50*100, 2),
                               "d_model": d_model, "n_layers": n_layers})

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    report_data = {
        "main": {
            "params": params,
            "best_acc": round(best_acc * 100, 2),
            "epoch_99": ep99,
            "history": history  # list of (epoch, loss, acc)
        },
        "sweep": sweep_results
    }
    with open("/workspace/hackathon/report_data.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("\n결과 저장 완료: report_data.json")

if __name__ == "__main__":
    main()
