# KRAFTON AI R&D HACKATHON

KRAFTON AI R&D 해커톤 참가 코드 저장소입니다.
예선(Preliminary)과 본선(Main) 2개 라운드로 구성되며, 각 라운드는 Day 1 / Day 2 두 문제로 진행되었습니다.

---

## 저장소 구조

```
hackathon/
├── preliminary/
│   ├── day1/
│   │   ├── [KRAFTON AI R&D HACKATHON] DAY1.html   # 문제지
│   │   ├── 1_1.py                                  # 문제 1-1: Zero-Parameter 트랜스포머 곱셈기
│   │   ├── 1_2.py                                  # 문제 1-2: 학습 기반 Causal Transformer 곱셈기
│   │   ├── 1_1.md                                  # 문제 1-1 설계 보고서
│   │   └── 1_2.md                                  # 문제 1-2 실험 결과 보고서
│   └── day2/
│       ├── [KRAFTON AI R&D HACKATHON] DAY2.html    # 문제지
│       ├── faster_RANSAC.py                         # Bitwise RANSAC (Numba JIT)
│       ├── faster_Gradient.py                       # Hybrid RANSAC + Gradient Descent
│       └── answer/problem2_answer_local_search.txt  # 제출 답안
└── main/
    ├── day1/
    │   ├── [KRAFTON AI R&D HACKATHON] DAY1 P2.html # 문제지
    │   ├── dataset.csv                              # 경기 데이터
    │   └── final_code.py                            # 플레이어 실력 예측 파이프라인
    └── day2/
        ├── KRAFTON_AI_HACKATHON_Finals_DAY2.html    # 문제지
        ├── video_pipeline.py                        # 2-Agent 비디오 분석 파이프라인
        ├── prompt_analysis.py                       # 프롬프트 → JSON 지시서 분해
        ├── run_all.py                               # 20개 영상 비동기 일괄 처리
        └── answer.txt                               # 제출 답안
```

---

## 예선 (Preliminary)

### Day 1 — 6비트 정수 곱셈기 (Transformer)

6비트 정수 두 개를 입력받아 12비트 곱을 출력하는 트랜스포머 모델을 설계하는 문제입니다.
두 가지 방향으로 접근했습니다.

#### 문제 1-1: Zero-Parameter 곱셈기 (`1_1.py`)

가중치 학습 없이 트랜스포머의 Attention과 MLP를 디지털 논리 게이트로 직접 설계했습니다.

| 항목 | 내용 |
|------|------|
| 아키텍처 | MicroMultiplier (2-Layer Hardcoded Transformer) |
| 학습 가능 파라미터 (P₁) | **최소화** (Zero-Parameter Routing) |
| 입출력 형식 | `[A₀..A₅, B₀..B₅]` → autoregressive `[P₀..P₁₁]` (LSB-First) |

**설계 원리**

- **Layer 1 (Partial Sum)**: 6개의 독립 어텐션 헤드가 $i+j=k$를 만족하는 $(A_i, B_j)$ 쌍을 하드코딩된 마스크로 라우팅. MLP가 `ReLU(2x - 1.5)` 로 AND 게이트를 구성해 부분곱을 산출
- **Layer 2 (Carry Deduction)**: 과거 출력 비트를 $2^{-(k-m)}$ 가중치로 참조해 자리올림을 역산. MLP의 교대합 `ReLU` 조합으로 `Total mod 2`(홀짝 판별)를 계산

#### 문제 1-2: 학습 기반 곱셈기 (`1_2.py`)

표준 Causal Transformer를 최소 파라미터로 학습시켜 99% 이상의 정확도를 달성했습니다.

| 항목 | 내용 |
|------|------|
| 아키텍처 | ConceptMultiplier (`d_model=32, n_heads=4, n_layers=3, d_ff=128`) |
| 학습 가능 파라미터 (P₂) | **39,010** |
| 최종 Val Accuracy | **99.05%** (80 에폭) |
| 학습 시간 | ~6분 (GPU) |

**아키텍처 선택 근거**

- Causal mask로 자기회귀 생성 시 carry propagation을 순차적으로 모방
- Learned Positional Embedding으로 비트 위치 간 매칭을 자유롭게 학습
- Weight Tying (임베딩 ↔ 출력 헤드)으로 파라미터 최소화

**파라미터 크기 탐색 결과**

| d_model | n_layers | 파라미터 수 | Acc@50ep |
|--------:|--------:|----------:|--------:|
| 8 | 2 | 1,426 | 10.78% |
| 16 | 2 | 7,010 | 15.35% |
| 16 | 3 | 10,290 | 46.63% |
| 24 | 3 | 22,346 | 81.72% |
| **32** | **3** | **39,010** | **86.63% → 99.05%@80ep** |

---

### Day 2 — LFSR Tap 역추적

알 수 없는 LFSR(Linear Feedback Shift Register)이 생성한 비트열에서 탭(tap) 위치를 역추적하고, 주어진 64비트 프리픽스로부터 192비트를 예측하는 문제입니다.

GF(2) 선형 방정식계 `X·w ≡ Y (mod 2)` 를 푸는 2단계 하이브리드 전략을 사용했습니다.

#### Phase 1 — Bitwise RANSAC (`faster_RANSAC.py`)

- 64개의 int8 특징을 1개의 uint64로 **비트 패킹**
- Numba `@njit` JIT 컴파일로 최대 1,500만 번 시도
- 매 시도마다 랜덤 샘플 64개로 GF(2) 가우스 소거법 수행
- **2단계 Early Rejection**: 100개 빠른 검증 → 통과 시 900개 추가 검증 → 전체 데이터 최종 확인

#### Phase 2 — Gradient Descent 검증 (`faster_Gradient.py`)

- RANSAC 결과를 **Warm-Start**로 사용하는 PyTorch 모델
- Sigmoid 마스크로 탭 위치를 연속 공간에서 최적화
- Trimmed Loss (상위 80% 샘플 유지)로 노이즈에 강인
- 최종 탭 확정 후 192비트 시퀀스 생성

---

## 본선 (Main)

### Day 1 — 플레이어 실력 예측

오염된 CSV 데이터에서 10명의 플레이어가 21일간 경기한 기록을 복구하고, 22~50일차의 누적 킬 수를 예측하는 문제입니다.

| 단계 | 기법 | 설명 |
|------|------|------|
| 1 | **Data Recovery** | 줄바꿈으로 오염된 CSV를 커스텀 파서로 복구 |
| 2 | **Bradley-Terry Model** | 승/패 기록에서 플레이어 실력(logit)을 MLE로 추정, L2 정규화 적용 |
| 3 | **TSP Optimization** | 날짜가 `?`로 가려진 블록들을 순열 탐색으로 올바른 시간 순서에 배치 |
| 4 | **Selective Damped Trend** | 최근 10일 선형회귀 + R² 검증(threshold=0.4), 감쇠 계수 φ=0.75로 미래 예측 |
| 5 | **Markov Chain Simulation** | Gauntlet 대결에서 예측 실력을 투입해 기대 킬 수를 확률적으로 계산 |

---

### Day 2 — 비디오 VQA 파이프라인

> **가장 난이도 높은 문제 — 최종 정확도 약 40점**

20개의 영상과 객관식 질문(A-Z)을 입력받아 정답을 자동으로 추론하는 문제입니다.
질문 유형은 브랜드/외관 식별, 장면 수 카운트, N번째 등장 항목 찾기 등 다양했습니다.

#### 시스템 구조

```
입력: video.mp4 + 객관식 질문
           │
           ▼
   [prompt_analysis.py]
   Gemini로 프롬프트를 JSON 지시서로 분해
   (task_category, scan_query, options 등)
           │
           ▼
   [TimestampAgent]
   1. 1초 간격 전체 프레임 추출 (ThreadPoolExecutor x8)
   2. 10×1 그리드 이미지 생성 (10초 단위)
   3. Gemini → 관련 시점(초) JSON 배열 추출
   4. 인접 시점 병합 → ffmpeg로 condensed.mp4 생성
           │
           ▼
   [AnswerAgent]
   1. Task 유형 자동 감지 (sequence / count / appearance)
   2. Gemini File API로 압축 영상 업로드
   3. Task별 CoT 5단계 프롬프트 추론 (thinking_budget=8192)
   4. "ANSWER: X" 파싱
           │
           ▼
      최종 정답 (A-Z)
```

#### Task별 CoT 전략

| Task | 감지 키워드 | 전략 |
|------|------------|------|
| `sequence` | fifth, 5th, 번째, 순서 | 등장 목록 작성 → 비활성 항목 제거 → N번째 특정 |
| `count` | how many, distinct, 몇 | 고유 항목 목록 → 중복 제거 → 개수 검증 |
| `appearance` | 그 외 | 관련 시점 탐색 → 시각 디테일 분석 → 선택지 대조 |

## 의존성

```bash
# 공통
pip install numpy pandas scipy scikit-learn matplotlib

# 예선 Day 1
pip install torch

# 예선 Day 2
pip install torch numba

# 본선 Day 2
pip install opencv-python pillow google-generativeai google-genai
apt-get install ffmpeg
```

## API 키 설정

```json
// secret/api_key.json  (git에 포함되지 않음)
{
  "gemini": "YOUR_GEMINI_API_KEY"
}
```
