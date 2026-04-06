import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import time

# ==========================================
# 1. 엔지니어님의 개념을 담은 아키텍처 DAY1.html]
# ==========================================
class ConceptMultiplier(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        # 1. 어휘 임베딩 (0, 1) DAY1.html]
        self.emb = nn.Embedding(2, d_model)
        
        # 2. 위치 인코딩: 고정(Fixed) 대신 모델이 특정 비트 위치(A_i, B_j)를 
        # 스스로 매칭(Routing)할 수 있도록 학습형(Learned)으로 변경
        self.pos_emb = nn.Embedding(24, d_model)
        
        # 3. Transformer Decoder (표준 순전파 및 셀프 어텐션) DAY1.html]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        
        # 파라미터 최소화를 위한 가중치 공유(Weight Tying) DAY1.html]
        self.head.weight = self.emb.weight

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        # 토큰 값 + 위치 정보 결합
        x_emb = self.emb(x) + self.pos_emb(positions).unsqueeze(0)
        
        # 자기회귀(Autoregressive) 생성을 위한 Causal Mask DAY1.html]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(x_emb, mask=mask, is_causal=True)
        return self.head(self.ln_f(out))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. 데이터셋 (LSB-First 고정 형식) DAY1.html]
# ==========================================
class MultiplierDataset(Dataset):
    def __init__(self, num_samples=100000):
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

# ==========================================
# 3. 고정 학습 프로토콜 (Fixed Training) DAY1.html]
# ==========================================
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 튜닝 포인트: d_model과 n_layers를 깎아가며 최소 파라미터(P_2)를 찾습니다.
    model = ConceptMultiplier(d_model=32, n_heads=4, n_layers=3, d_ff=128).to(device)
    p2_count = count_parameters(model)
    print(f"[{device.type.upper()}] 모델 초기화 완료. Trainable Parameters (P_2): {p2_count}")
    
    print("데이터셋 생성 중...")
    train_loader = DataLoader(MultiplierDataset(100000), batch_size=256, shuffle=True)
    test_dataset = MultiplierDataset(10000)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # DAY1.html]
    scheduler = CosineAnnealingLR(optimizer, T_max=200) # DAY1.html]
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(200):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            # x: [A0..A5, B0..B5, P0..P10], y: [A1..A5, B0..B5, P0..P11]
            x, y = batch[:, :-1], batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Loss: 12개 출력 토큰 위치에 대해서만 적용 (index 11부터 끝까지) DAY1.html]
            loss = criterion(logits[:, 11:, :].reshape(-1, 2), y[:, 11:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # 10 에폭마다 검증 수행
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    x, y = batch[:, :-1], batch[:, 1:]
                    logits = model(x)
                    preds = logits[:, 11:, :].argmax(dim=-1)
                    # 12비트가 모두 일치해야 정답
                    correct += (preds == y[:, 11:]).all(dim=1).sum().item()
            
            acc = correct / 10000
            print(f"Epoch {epoch+1:03d}/200 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.2f}%")
            
            # 99% 달성 시 조기 종료 알림 (대회 제출은 200에폭 기준) DAY1.html]
            if acc >= 0.99:
                print(f"🎉 99% 달성! 이 모델 사이즈로 제출 가능합니다.")

    print(f"\n학습 종료. 소요 시간: {(time.time() - start_time)/60:.2f}분")

if __name__ == '__main__':
    train_and_evaluate()