import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. 데이터 전처리: 커스텀 파서 (Data Recovery)
# ==========================================
def load_and_fix_data(file_path):
    print("[*] 오염된 CSV 데이터를 분석하고 복구하며 로드합니다...")
    parsed_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    current_row = None
    fixed_count = 0

    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line: continue
        
        first_col = line.split(',')[0]
        if first_col.isdigit() or first_col == '?':
            if current_row: parsed_data.append(current_row)
            current_row = line.split(',', 2)
        else:
            if current_row:
                current_row[2] += " " + line
                fixed_count += 1
    if current_row: parsed_data.append(current_row)
    print(f"[*] 데이터 복구 완료! 총 {fixed_count}개의 유실 데이터를 수정했습니다.")
    return pd.DataFrame(parsed_data, columns=header)

# ==========================================
# 2. 실력 추정: Bradley-Terry 모델
# ==========================================
def extract_win_matrix(events_list, num_players=10):
    matrix = np.zeros((num_players, num_players))
    for event_str in events_list:
        if not isinstance(event_str, str) or '>' not in event_str: continue
        for k in event_str.strip().split():
            winner, loser = map(int, k.split('>'))
            matrix[winner, loser] += 1
    return matrix

def fit_bradley_terry(wins_matrix, num_players=10):
    def nll(skills):
        loss = 0
        for i in range(num_players):
            for j in range(num_players):
                if i != j and wins_matrix[i, j] > 0:
                    prob = expit(skills[i] - skills[j])
                    loss -= wins_matrix[i, j] * np.log(prob + 1e-10)
        loss += 0.01 * np.sum(skills**2) # L2 Regularization
        return loss
    res = minimize(nll, np.zeros(num_players), method='L-BFGS-B')
    return res.x

# ==========================================
# 3. 과거 복원: TSP Optimization
# ==========================================
def sort_blocks_tsp(start_skill, end_skill, blocks):
    block_skills = [fit_bradley_terry(extract_win_matrix(b)) for b in blocks]
    best_cost = float('inf')
    best_order = None
    
    for perm in itertools.permutations(range(len(blocks))):
        current_perm_skills = [block_skills[i] for i in perm]
        # 시작점 -> 첫 블록 -> ... -> 마지막 블록 -> 도착점 거리 합 최소화
        cost = np.sum((start_skill - current_perm_skills[0])**2)
        for i in range(len(current_perm_skills) - 1):
            cost += np.sum((current_perm_skills[i] - current_perm_skills[i+1])**2)
        cost += np.sum((current_perm_skills[-1] - end_skill)**2)
        
        if cost < best_cost:
            best_cost = cost
            best_order = current_perm_skills
    return list(best_order)

# ==========================================
# 4. 확률 시뮬레이션: Markov Chain
# ==========================================
def get_expected_kills(gauntlet, skill_vector):
    kills = {p: 0.0 for p in gauntlet}
    def win_prob(pA, pB): return expit(skill_vector[pA] - skill_vector[pB])
    
    # 1, 2번 타자 초기 대결
    alive_probs = {
        gauntlet[0]: win_prob(gauntlet[0], gauntlet[1]),
        gauntlet[1]: win_prob(gauntlet[1], gauntlet[0])
    }
    kills[gauntlet[0]] += alive_probs[gauntlet[0]]
    kills[gauntlet[1]] += alive_probs[gauntlet[1]]
    
    # 3, 4, 5번 타자 순차 투입
    for i in range(2, 5):
        next_challenger = gauntlet[i]
        new_alive_probs = {next_challenger: 0.0}
        for survivor, p_surv_is_here in alive_probs.items():
            p_s_wins = win_prob(survivor, next_challenger)
            p_c_wins = 1.0 - p_s_wins
            
            kills[survivor] += p_surv_is_here * p_s_wins
            kills[next_challenger] += p_surv_is_here * p_c_wins
            
            new_alive_probs[survivor] = new_alive_probs.get(survivor, 0.0) + p_surv_is_here * p_s_wins
            new_alive_probs[next_challenger] += p_surv_is_here * p_c_wins
        alive_probs = new_alive_probs
    return kills

# ==========================================
# 5. 메인 실행 파이프라인
# ==========================================
# 데이터 로드
df = load_and_fix_data('dataset.csv')

# 앵커 데이터 (1, 11, 21일차) 실력 추출
print("[*] Anchor days (1, 11, 21) 실력 분석 중...")
known_days = [1, 11, 21]
known_skills = {d: fit_bradley_terry(extract_win_matrix(df[df['day'] == str(d)]['events'].tolist())) for d in known_days}

# TSP 기반 가려진 날짜(?) 복원
print("[*] TSP를 이용한 시공간 복원(History Reconstruction) 시작...")
q_rows = df[df['day'] == '?']['events'].tolist()
blocks_2_10 = [q_rows[i*50 : (i+1)*50] for i in range(9)]
blocks_12_20 = [q_rows[i*50 : (i+1)*50] for i in range(9, 18)]

sorted_2_10 = sort_blocks_tsp(known_skills[1], known_skills[11], blocks_2_10)
sorted_12_20 = sort_blocks_tsp(known_skills[11], known_skills[21], blocks_12_20)

# 전체 과거 실력 데이터 통합 (1~21일)
all_skills = {1: known_skills[1], 11: known_skills[11], 21: known_skills[21]}
all_skills.update({day: sorted_2_10[i] for i, day in enumerate(range(2, 11))})
all_skills.update({day: sorted_12_20[i] for i, day in enumerate(range(12, 21))})

# ------------------------------------------
# 미래 예측 엔진: Selective Damped Trend
# ------------------------------------------
print("\n[*] 미래 예측 엔진 가동: Selective Damped Trend (phi=0.75, R2_threshold=0.4)")
skills_matrix = np.array([all_skills[d] for d in range(1, 22)])
days_axis = np.array(range(1, 22)).reshape(-1, 1)

phi = 0.75
r2_threshold = 0.4
effective_slopes = np.zeros(10)
r2_values = []

# 각 플레이어별 기울기 신뢰도($R^2$) 검증
for p in range(10):
    lr = LinearRegression()
    lr.fit(days_axis[-10:], skills_matrix[-10:, p]) # 최근 10일 집중 분석
    r2 = lr.score(days_axis[-10:], skills_matrix[-10:, p])
    r2_values.append(r2)
    
    if r2 >= r2_threshold:
        effective_slopes[p] = lr.coef_[0]
        status = "Trend"
    else:
        effective_slopes[p] = 0.0 # 불확실한 경우 Static(정지) 처리
        status = "Static"
    print(f"  - User {p} | R2: {r2:.3f} | Slope: {effective_slopes[p]:.4f} | Status: {status}")

# 22~50일차 동적 실력 생성
predicted_skills = {}
curr_s = skills_matrix[-1].copy()
for d in range(22, 51):
    h = d - 21
    curr_s += (phi ** h) * effective_slopes
    predicted_skills[d] = curr_s.copy()

# ------------------------------------------
# 최종 킬 수 시뮬레이션 (Markov 기반)
# ------------------------------------------
test_df = df[pd.to_numeric(df['day'], errors='coerce') >= 22]
final_kills = {i: 0.0 for i in range(10)}

for _, row in test_df.iterrows():
    day = int(row['day'])
    gauntlet = list(map(int, str(row['events']).split()))
    # 매일 변하는 실력을 마르코프 체인에 주입
    match_kills = get_expected_kills(gauntlet, predicted_skills[day])
    for p, k in match_kills.items():
        final_kills[p] += k

# ==========================================
# 6. 최종 결과 출력 및 시각화
# ==========================================
print("\n" + "="*50)
print("🎯 [FINAL RESULTS] 플레이어별 예측 누적 킬 수")
print("="*50)
for p in range(10):
    print(f"user_{p} : {final_kills[p]:.2f}")
print(f"Total Check: {sum(final_kills.values()):.2f} (Target: 5800)")

def plot_final_forecast(all_skills, slopes, r2_values, phi, r2_threshold):
    plt.figure(figsize=(15, 8))
    past_days = np.array(range(1, 22))
    future_days = np.array(range(22, 51))
    
    for p in range(10):
        past_values = [all_skills[d][p] for d in past_days]
        color = plt.plot(past_days, past_values, label=f'User {p} (R2:{r2_values[p]:.2f})', lw=2)[0].get_color()
        
        future_values = []
        temp_s = past_values[-1]
        for d in future_days:
            h = d - 21
            temp_s += (phi ** h) * slopes[p]
            future_values.append(temp_s)
        
        plt.plot(future_days, future_values, linestyle='--', color=color, alpha=0.8)
        if r2_values[p] < r2_threshold:
            plt.text(50.5, future_values[-1], " (Static)", color=color, fontsize=8)

    plt.axvline(x=21, color='black', alpha=0.5)
    plt.title(f"Final Skill Forecast (phi={phi}, R2_threshold={r2_threshold})")
    plt.xlabel("Day"); plt.ylabel("Skill (Logit)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, ls=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("future_skill_forecast.png", dpi=300)
    print("\n[*] 시각화 완료: 'future_skill_forecast.png' 저장됨.")

plot_final_forecast(all_skills, effective_slopes, r2_values, phi, r2_threshold)

# [기존 로직] 데이터 로더, BT 모델, TSP 복원, 마르코프 체인
# ==========================================
def load_and_fix_data(file_path):
    parsed_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    current_row = None
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line: continue
        first_col = line.split(',')[0]
        if first_col.isdigit() or first_col == '?':
            if current_row: parsed_data.append(current_row)
            current_row = line.split(',', 2)
        else:
            if current_row: current_row[2] += " " + line
    if current_row: parsed_data.append(current_row)
    return pd.DataFrame(parsed_data, columns=header)

def extract_win_matrix(events_list, num_players=10):
    matrix = np.zeros((num_players, num_players))
    for event_str in events_list:
        if not isinstance(event_str, str) or '>' not in event_str: continue
        for k in event_str.strip().split():
            winner, loser = map(int, k.split('>'))
            matrix[winner, loser] += 1
    return matrix

def fit_bradley_terry(wins_matrix, num_players=10):
    def nll(skills):
        loss = 0
        for i in range(num_players):
            for j in range(num_players):
                if i != j and wins_matrix[i, j] > 0:
                    prob = expit(skills[i] - skills[j])
                    loss -= wins_matrix[i, j] * np.log(prob + 1e-10)
        loss += 0.01 * np.sum(skills**2)
        return loss
    return minimize(nll, np.zeros(num_players), method='L-BFGS-B').x

def sort_blocks_tsp(start_skill, end_skill, blocks):
    block_skills = [fit_bradley_terry(extract_win_matrix(b)) for b in blocks]
    best_cost, best_order = float('inf'), None
    for perm in itertools.permutations(range(len(blocks))):
        current_perm_skills = [block_skills[i] for i in perm]
        cost = np.sum((start_skill - current_perm_skills[0])**2)
        for i in range(len(current_perm_skills) - 1):
            cost += np.sum((current_perm_skills[i] - current_perm_skills[i+1])**2)
        cost += np.sum((current_perm_skills[-1] - end_skill)**2)
        if cost < best_cost: best_cost, best_order = cost, current_perm_skills
    return list(best_order)

def get_expected_kills(gauntlet, skill_vector):
    kills = {p: 0.0 for p in gauntlet}
    def win_prob(pA, pB): return expit(skill_vector[pA] - skill_vector[pB])
    alive_probs = {gauntlet[0]: win_prob(gauntlet[0], gauntlet[1]), gauntlet[1]: win_prob(gauntlet[1], gauntlet[0])}
    kills[gauntlet[0]] += alive_probs[gauntlet[0]]; kills[gauntlet[1]] += alive_probs[gauntlet[1]]
    for i in range(2, 5):
        next_c = gauntlet[i]; new_alive = {next_c: 0.0}
        for surv, p_h in alive_probs.items():
            p_s_w = win_prob(surv, next_c)
            kills[surv] += p_h * p_s_w; kills[next_c] += p_h * (1-p_s_w)
            new_alive[surv] = new_alive.get(surv, 0.0) + p_h * p_s_w
            new_alive[next_c] += p_h * (1-p_s_w)
        alive_probs = new_alive
    return kills

# ==========================================
# 5. 메인 로직 실행 및 확장 검증
# ==========================================
df = load_and_fix_data('dataset.csv')
known_skills = {d: fit_bradley_terry(extract_win_matrix(df[df['day'] == str(d)]['events'].tolist())) for d in [1, 11, 21]}

q_rows = df[df['day'] == '?']['events'].tolist()
blocks_2_10 = [q_rows[i*50 : (i+1)*50] for i in range(9)]
blocks_12_20 = [q_rows[i*50 : (i+1)*50] for i in range(9, 18)]

sorted_2_10 = sort_blocks_tsp(known_skills[1], known_skills[11], blocks_2_10)
sorted_12_20 = sort_blocks_tsp(known_skills[11], known_skills[21], blocks_12_20)

all_skills = {1: known_skills[1], 11: known_skills[11], 21: known_skills[21]}
all_skills.update({day: sorted_2_10[i] for i, day in enumerate(range(2, 11))})
all_skills.update({day: sorted_12_20[i] for i, day in enumerate(range(12, 21))})

skills_matrix = np.array([all_skills[d] for d in range(1, 22)])
days_axis = np.array(range(1, 22)).reshape(-1, 1)

# [검증 1] Backtesting: 1~11일차로 12~21일차 예측해보기
print("\n" + "-"*30 + "\n[검증 1] Backtesting (12~21일차 예측 오차)")
temp_slopes = np.zeros(10)
for p in range(10):
    lr = LinearRegression().fit(days_axis[:11], skills_matrix[:11, p])
    temp_slopes[p] = lr.coef_[0] if lr.score(days_axis[:11], skills_matrix[:11, p]) > 0.4 else 0
backtest_error = np.mean([np.abs(all_skills[21][p] - (all_skills[11][p] + temp_slopes[p]*10)) for p in range(10)])
print(f"  - 평균 실력 예측 오차 (MAE): {backtest_error:.4f}")

# 미래 예측 파라미터
phi = 0.75
r2_threshold = 0.4
effective_slopes = np.zeros(10)
r2_values = []

print("\n" + "-"*30 + "\n[검증 2] $R^2$ 신뢰도 및 상태 분석")
for p in range(10):
    lr = LinearRegression().fit(days_axis[-10:], skills_matrix[-10:, p])
    r2 = lr.score(days_axis[-10:], skills_matrix[-10:, p])
    r2_values.append(r2)
    effective_slopes[p] = lr.coef_[0] if r2 >= r2_threshold else 0.0

# 미래 실력 생성 및 킬 수 계산
predicted_skills = {}
curr_s = skills_matrix[-1].copy()
for d in range(22, 51):
    h = d - 21
    curr_s += (phi ** h) * effective_slopes
    predicted_skills[d] = curr_s.copy()

test_df = df[pd.to_numeric(df['day'], errors='coerce') >= 22]
final_kills = {i: 0.0 for i in range(10)}
for _, row in test_df.iterrows():
    match_k = get_expected_kills(list(map(int, str(row['events']).split())), predicted_skills[int(row['day'])])
    for p, k in match_k.items(): final_kills[p] += k

# 최종 테이블 출력
print("\n🎯 [최종 결과 리포트]")
print(f"{'Player':<8} | {'Kills':<10} | {'R2':<8} | {'Status':<8}")
print("-" * 45)
for p in range(10):
    status = "Trend" if r2_values[p] >= r2_threshold else "Static"
    print(f"user_{p:<3} | {final_kills[p]:<10.2f} | {r2_values[p]:<8.3f} | {status}")
print(f"Total: {sum(final_kills.values()):.2f}")

# [검증 3] Win-Rate Sanity Check: 50일차 User 9의 지배력 확인
print("\n" + "-"*30 + "\n[검증 3] 50일차 승률 안정성 테스트 (vs User 9)")
day50_s = predicted_skills[50]
for i in [0, 4, 8]: # 대표 유저 비교
    prob = expit(day50_s[9] - day50_s[i])
    print(f"  - User 9 vs User {i} 승률: {prob*100:.2f}%")
