import numpy as np
from numba import njit
import time

# 1. 비트 패킹: 64개의 int8 데이터를 1개의 uint64 정수로 압축
def pack_X(X):
    N = X.shape[0]
    X_packed = np.zeros(N, dtype=np.uint64)
    for i in range(64):
        X_packed |= (X[:, i].astype(np.uint64) << np.uint64(i))
    return X_packed

# 2. 비트 단위 초고속 GF(2) 가우스 소거법
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
            t_A = A[row]
            A[row] = A[pivot]
            A[pivot] = t_A
            t_y = y[row]
            y[row] = y[pivot]
            y[pivot] = t_y
            
        # Eliminate
        for r in range(n):
            if r != row and (A[r] & mask):
                A[r] ^= A[row]
                y[r] ^= y[row]
        row += 1
        
    if row < 64: return False
    
    # 정답 추출
    for r in range(64):
        w_out[r] = y[r]
    return True

# 3. 궁극의 Bitwise RANSAC (최대 1,500만 번 시도 -> 99.99% 성공 보장)
@njit
def run_ransac_bitwise(X_packed, Y, max_trials=15000000):
    N = X_packed.shape[0]
    W = 64
    
    A_sub = np.zeros(W, dtype=np.uint64)
    y_sub = np.zeros(W, dtype=np.uint8)
    w_out = np.zeros(W, dtype=np.int8)
    
    for trial in range(1, max_trials + 1):
        # 빠른 샘플링
        for i in range(W):
            r_idx = np.random.randint(0, N)
            A_sub[i] = X_packed[r_idx]
            y_sub[i] = Y[r_idx]
            
        if not solve_gf2_bitwise(A_sub, y_sub, w_out):
            continue
            
        # 가중치 배열을 uint64 1개로 압축
        w_packed = np.uint64(0)
        for i in range(W):
            if w_out[i]:
                w_packed |= (np.uint64(1) << np.uint64(i))
                
        # 1차 초고속 조기 검증 (100개)
        correct = 0
        for i in range(100):
            v_id = np.random.randint(0, N)
            overlap = X_packed[v_id] & w_packed
            # Popcount (비트가 1인 개수 세기)
            c = 0
            while overlap:
                overlap &= overlap - np.uint64(1)
                c += 1
            if (c % 2) == Y[v_id]:
                correct += 1
                
        if correct < 65: continue # 가차없이 버림
            
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
                
        if correct > 720:
            # 최종 전체 검증
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

if __name__ == "__main__":
    DATA_PATH = "DAY2_data.txt" 
    W = 64
    
    print("데이터 로드 중...")
    try:
        with open(DATA_PATH, 'r') as f:
            lines = f.read().splitlines()
    except Exception as e:
        print(f"파일 오류: {e}")
        exit()
        
    X_list, Y_list = [], []
    for line in lines:
        if len(line) < 256: continue
        seq = [int(c) for c in line]
        for n in range(W, 256):
            X_list.append(seq[n - W : n])
            Y_list.append(seq[n])
            
    X = np.array(X_list, dtype=np.int8)
    Y = np.array(Y_list, dtype=np.uint8)
    
    print("데이터를 64비트 정수로 패킹 중 (Bit-packing)...")
    X_packed = pack_X(X)
    
    print(f"총 {len(X_packed)}개의 데이터 준비 완료.")
    print("🚀 궁극의 Bitwise RANSAC 탐색 시작! (최대 1,500만 번 시도 보장)")
    start_time = time.time()
    
    # 1,500만 번을 돌려도 비트 연산이라 금방 끝납니다.
    trial, w_pred, final_acc = run_ransac_bitwise(X_packed, Y, max_trials=15000000)
    
    if trial != -1:
        found_taps = sorted([W - i for i, val in enumerate(w_pred) if val == 1])
        print(f"\n[!] 🎯 정답 오프셋 확정! (Trial {trial:,d})")
        print(f"최종 정확도: {final_acc*100:.2f}%")
        print(f"추출된 오프셋 Taps (d_i): {found_taps}")
        print(f"소요 시간: {time.time() - start_time:.2f}초")
        
        # 3. Test Sequence 생성
        test_prefix = "0000010100011010010101100101001110100011110010110011010000111010"
        seq = [int(x) for x in test_prefix]
        
        for _ in range(192):
            n = len(seq)
            next_bit = 0
            for d in found_taps:
                if n - d >= 0:
                    next_bit ^= seq[n - d]
            seq.append(next_bit)
            
        answer = "".join(map(str, seq[64:]))
        print(f"\n[Final Submission]")
        print(f"Answer (192 bits):\n{answer}")
    else:
        print(f"최대 시도 횟수(1,500만 번) 초과. (소요 시간: {time.time() - start_time:.2f}초)")