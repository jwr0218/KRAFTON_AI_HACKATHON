import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numba import njit
import time

# =====================================================================
# [Phase 1] Ultra-Fast Bitwise RANSAC (초고속 통계적 탐색)
# =====================================================================
def pack_X(X):
    """ 64개의 int8 데이터를 1개의 uint64 정수로 압축 """
    N = X.shape[0]
    X_packed = np.zeros(N, dtype=np.uint64)
    for i in range(64):
        X_packed |= (X[:, i].astype(np.uint64) << np.uint64(i))
    return X_packed

@njit
def solve_gf2_bitwise(A_sub, y_sub, w_out):
    A = A_sub.copy()
    y = y_sub.copy()
    n = 64
    row = 0
    
    for col in range(64):
        mask = np.uint64(1) << np.uint64(col)
        pivot = -1
        for r in range(row, n):
            if A[r] & mask:
                pivot = r
                break
        if pivot == -1: return False
        
        # Swap
        if pivot != row:
            t_A = A[row]; A[row] = A[pivot]; A[pivot] = t_A
            t_y = y[row]; y[row] = y[pivot]; y[pivot] = t_y
            
        # Eliminate
        for r in range(n):
            if r != row and (A[r] & mask):
                A[r] ^= A[row]
                y[r] ^= y[row]
        row += 1
        
    if row < 64: return False
    
    for r in range(64):
        w_out[r] = y[r]
    return True

@njit
def run_ransac_bitwise(X_packed, Y, max_trials=15000000):
    N = X_packed.shape[0]
    W = 64
    
    A_sub = np.zeros(W, dtype=np.uint64)
    y_sub = np.zeros(W, dtype=np.uint8)
    w_out = np.zeros(W, dtype=np.int8)
    
    best_acc = 0.0
    
    for trial in range(1, max_trials + 1):
        for i in range(W):
            r_idx = np.random.randint(0, N)
            A_sub[i] = X_packed[r_idx]
            y_sub[i] = Y[r_idx]
            
        if not solve_gf2_bitwise(A_sub, y_sub, w_out):
            continue
            
        w_packed = np.uint64(0)
        for i in range(W):
            if w_out[i]:
                w_packed |= (np.uint64(1) << np.uint64(i))
                
        # 1차 초고속 조기 검증 (100개)
        correct = 0
        for i in range(100):
            v_id = np.random.randint(0, N)
            overlap = X_packed[v_id] & w_packed
            c = 0
            while overlap:
                overlap &= overlap - np.uint64(1)
                c += 1
            if (c % 2) == Y[v_id]:
                correct += 1
                
        if correct < 65: continue
            
        # 2차 검증 (900개)
        for i in range(100, 1000):
            v_id = np.random.randint(0, N)
            overlap = X_packed[v_id] & w_packed
            c = 0
            while overlap:
                overlap &= overlap - np.uint64(1)
                c += 1
            if (c % 2) == Y[v_id]:
                correct += 1
                
        acc_val = correct / 1000.0
        
        # Numba 지원 로그 출력
        if acc_val > best_acc:
            best_acc = acc_val
            if acc_val > 0.65:
                print(" ✨ [Trial", trial, "] 새로운 후보 발견! | 정확도:", acc_val*100, "%")
                
        if trial % 1000000 == 0:
            print(" ⏳ [Status]", trial, "회 탐색 중... (Best:", best_acc*100, "%)")
                
        if acc_val > 0.72:
            full_correct = 0
            for i in range(N):
                overlap = X_packed[i] & w_packed
                c = 0
                while overlap:
                    overlap &= overlap - np.uint64(1)
                    c += 1
                if (c % 2) == Y[i]:
                    full_correct += 1
                    
            final_acc = full_correct / N
            if final_acc > 0.78:
                return trial, w_out, final_acc
                
    return -1, w_out, 0.0

# =====================================================================
# [Phase 2] Gradient Descent (수치적 검증 및 파인튜닝)
# =====================================================================
class WarmStartProductXOR(nn.Module):
    def __init__(self, W=64, prior_taps=None):
        super().__init__()
        # 모든 마스크를 -5.0(비활성화)으로 초기화하여 곱셈 기울기 소실 방지
        init_logits = torch.ones(1, W) * (-5.0) 
        
        if prior_taps:
            for tap in prior_taps:
                idx = W - tap
                if 0 <= idx < W:
                    # RANSAC이 알려준 힌트 위치만 +5.0(활성화)으로 편향 (Warm-Start)
                    init_logits[0, idx] = 5.0
                    
        self.mask_logits = nn.Parameter(init_logits)

    def forward(self, x):
        x_p = 1.0 - 2.0 * x 
        mask = torch.sigmoid(self.mask_logits)
        masked_x = mask * x_p + (1.0 - mask)
        return torch.prod(masked_x, dim=1), mask

def train_hybrid_gd(X_list, Y_list, prior_taps, W=64, epochs=100):
    print("\n" + "="*50)
    print("🧠 [Phase 2] GD 수치적 파인튜닝 & 검증 (Trimmed Loss)")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 딥러닝 검증은 속도를 위해 2000개 샘플만 사용해도 충분합니다.
    X = torch.tensor(X_list[:2000], dtype=torch.float32).to(device)
    Y_p = (1.0 - 2.0 * torch.tensor(Y_list[:2000], dtype=torch.float32)).to(device)
    k_clean = int(X.shape[0] * 0.8) # 상위 20% 노이즈 절사 기준
    
    model = WarmStartProductXOR(W=W, prior_taps=prior_taps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred, mask = model(X)
        
        loss_per_sample = -(pred * Y_p.unsqueeze(0))
        sorted_loss, _ = torch.sort(loss_per_sample, dim=1)
        clean_loss = torch.mean(sorted_loss[:, :k_clean], dim=1)
        
        l1_loss = 0.02 * torch.sum(mask, dim=1)
        (clean_loss + l1_loss).backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f" 📈 [Epoch {epoch:3d}/{epochs}] Clean Loss: {clean_loss.item():.4f} (목표: -1.0)")

    best_mask = torch.sigmoid(model.mask_logits).detach().cpu().numpy()[0]
    return sorted([W - i for i, val in enumerate(best_mask) if val > 0.5])

# =====================================================================
# [Phase 3] 192비트 예측
# =====================================================================
def generate_answer(taps, prefix_str, num_to_predict=192):
    print("\n" + "="*50)
    print("🔮 [Phase 3] 최종 192비트 예측 시퀀스 생성")
    print("="*50)
    
    seq = [int(c) for c in prefix_str]
    for _ in range(num_to_predict):
        n = len(seq)
        next_bit = 0
        for d in taps:
            if n - d >= 0:
                next_bit ^= seq[n - d]
        seq.append(next_bit)
    
    return "".join(map(str, seq[len(prefix_str):]))

# =====================================================================
# Main Execution
# =====================================================================
def main():
    DATA_PATH = "DAY2_data.txt"
    TEST_PREFIX = "0000010100011010010101100101001110100011110010110011010000111010"
    W = 64
    
    print("데이터 로드 중...")
    try:
        with open(DATA_PATH, 'r') as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}"); return
        
    X_list, Y_list = [], []
    for line in lines:
        if len(line) < 256: continue
        seq = [int(c) for c in line]
        for n in range(W, 256):
            X_list.append(seq[n - W : n])
            Y_list.append(seq[n])
            
    print("데이터를 64비트 정수로 패킹 중 (Bit-packing)...")
    X = np.array(X_list, dtype=np.int8)
    Y = np.array(Y_list, dtype=np.uint8)
    X_packed = pack_X(X)
    
    print(f"\n🚀 [Phase 1] 궁극의 Bitwise RANSAC 탐색 시작! (최대 1,500만 번)")
    start_time = time.time()
    
    trial, w_pred, final_acc = run_ransac_bitwise(X_packed, Y, max_trials=15000000)
    
    if trial != -1:
        # 힌트 추출
        hints = sorted([W - i for i, val in enumerate(w_pred) if val == 1])
        print(f"\n💡 RANSAC이 찾은 후보 힌트: {hints} (정확도: {final_acc*100:.2f}%)")
        print(f"⏱️ RANSAC 소요 시간: {time.time() - start_time:.2f}초")
        
        # [2] GD 최적화 모델에 힌트를 주어 수치적 확정
        final_taps = train_hybrid_gd(X_list, Y_list, prior_taps=hints, W=W)
        print(f"\n✅ GD 모델이 확정한 최종 오프셋: {final_taps}")
        
        # [3] 192비트 예측 결과 생성
        answer_str = generate_answer(final_taps, TEST_PREFIX)
        
        print("\n" + "🏁" * 20)
        print("🚀 [최종 제출 정답 - 192 bits]")
        print(answer_str)
        print("🏁" * 20 + "\n")
    else:
        print(f"\n❌ 최대 시도 횟수 초과. 정답을 찾지 못했습니다.")

if __name__ == "__main__":
    main()