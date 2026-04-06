# Crafton Hackathon

Crafton 해커톤 참가 코드 저장소입니다. 예선(Preliminary)과 본선(Main) 2개 라운드로 구성되며, 각 라운드는 Day 1 / Day 2 두 문제로 진행되었습니다.

---

## 저장소 구조

```
hackathon/
├── preliminary/
│   ├── day1/final_code.py          # 예선 Day 1 — 플레이어 실력 예측
│   └── day2/
│       ├── faster_RANSAC.py        # 예선 Day 2 — Bitwise RANSAC
│       └── faster_Gradient.py      # 예선 Day 2 — Hybrid RANSAC + GD
└── main/
    ├── day1/final_code.py          # 본선 Day 1 — 플레이어 실력 예측
    └── day2/
        ├── video_pipeline.py       # 본선 Day 2 — 비디오 분석 파이프라인
        ├── prompt_analysis.py      # 프롬프트 → JSON 지시서 분해
        ├── run_all.py              # 20개 영상 비동기 일괄 처리
        └── PIPELINE_REPORT.md      # 파이프라인 설계 문서
```

---

## Day 1 — 플레이어 실력 예측 (예선 / 본선 공통)

### 문제 정의

오염된 CSV 데이터에서 10명의 플레이어가 21일간 경기한 기록을 복구하고, 22~50일차의 누적 킬 수를 예측합니다.

### 접근 방법

| 단계 | 기법 | 설명 |
|------|------|------|
| 1 | **Data Recovery** | 줄바꿈으로 오염된 CSV를 커스텀 파서로 복구 |
| 2 | **Bradley-Terry Model** | 플레이어 간 승/패 기록에서 실력(logit)을 MLE로 추정, L2 정규화 적용 |
| 3 | **TSP Optimization** | 날짜 정보가 `?`로 가려진 블록들을 순열 탐색으로 올바른 시간 순서에 배치 |
| 4 | **Selective Damped Trend** | 최근 10일 선형회귀 + R² 신뢰도 검증(threshold=0.4) → 불확실 플레이어는 Static 처리, 감쇠 계수 φ=0.75로 미래 실력 예측 |
| 5 | **Markov Chain Simulation** | Gauntlet 방식 대결에서 예측 실력을 투입해 기대 킬 수를 확률적으로 계산 |

### 핵심 알고리즘

```python
# Bradley-Terry: 실력 차이 기반 승률 추정
prob = sigmoid(skill_i - skill_j)

# Damped Trend: 감쇠 예측
skill_day_d = skill_day_21 + sum(phi^h * slope for h in range(1, d-21+1))

# Markov Chain: alive_probs 상태 전이
new_alive[survivor] += alive_probs[survivor] * win_prob(survivor, challenger)
```

---

## Day 2 (예선) — LFSR Tap 역추적

### 문제 정의

알 수 없는 Linear Feedback Shift Register(LFSR)가 생성한 비트열 데이터에서 탭(tap) 위치를 역추적하고, 주어진 64비트 프리픽스로부터 192비트를 예측합니다.

### 접근 방법

GF(2) 선형 방정식계 `X·w ≡ Y (mod 2)` 를 풀어 탭을 찾는 2단계 하이브리드 전략을 사용합니다.

#### Phase 1 — Bitwise RANSAC (`faster_RANSAC.py` / `faster_Gradient.py`)

- 64개의 int8 특징을 1개의 uint64로 **비트 패킹**
- Numba `@njit` JIT 컴파일로 최대 1,500만 번 시도
- 매 시도마다 랜덤 샘플 64개로 GF(2) 가우스 소거법 수행
- **2단계 Early Rejection**: 100개 빠른 검증 → 통과 시 900개 추가 검증 → 전체 데이터 최종 확인

#### Phase 2 — Gradient Descent 검증 (`faster_Gradient.py`)

- RANSAC 결과를 **Warm-Start**로 사용하는 PyTorch 모델
- Sigmoid 마스크로 탭 위치를 연속 공간에서 최적화
- Trimmed Loss (하위 80% 샘플만 사용)로 노이즈에 강인
- 최종 탭 위치가 확정되면 192비트 시퀀스 생성

---

## Day 2 (본선) — 비디오 VQA 파이프라인

> **가장 난이도 높은 문제 — 최종 정확도 약 40점**

### 문제 정의

20개의 영상과 객관식 질문을 입력받아 정답 알파벳(A-Z)을 자동으로 추론합니다.

### 시스템 구조

```
입력: video.mp4 + prompt 텍스트
           │
           ▼
   [prompt_analysis.py]
   Gemini로 프롬프트를 JSON 지시서로 분해
   (task_category, anchor_event, scan_query 등)
           │
           ▼
   [VideoPipeline]
           │
           ├─── [TimestampAgent]
           │      1. 1초 간격 전체 프레임 추출 (ThreadPoolExecutor)
           │      2. 10×1 그리드 이미지 생성 (10초 단위)
           │      3. Gemini → 관련 시점(초) JSON 배열 추출
           │      4. 인접 시점 병합 → 구간(segments) 생성
           │      5. ffmpeg → 구간만 잘라 condensed.mp4 생성
           │
           └─── [AnswerAgent]
                  1. Task 유형 자동 감지 (sequence / count / appearance)
                  2. Gemini File API로 압축 영상 업로드
                  3. Task별 CoT 5단계 프롬프트로 추론
                     (ThinkingConfig thinking_budget=8192)
                  4. "ANSWER: X" 패턴 파싱
                  │
                  ▼
             최종 정답 (A-Z)
```

### Task별 CoT 전략

| Task | 감지 키워드 | 전략 |
|------|------------|------|
| `sequence` | fifth, 5th, 번째, 순서 | 등장 목록 작성 → 비활성 항목 제거 → N번째 항목 특정 |
| `count` | how many, distinct, 몇 | 고유 항목 목록 → 중복 제거 → 개수 검증 |
| `appearance` | 그 외 | 관련 시점 탐색 → 시각 디테일 분석 → 선택지 대조 |

### 기술 스택

| 항목 | 내용 |
|------|------|
| **AI 모델** | Gemini (google-genai SDK), ThinkingConfig 활성화 |
| **프레임 추출** | OpenCV + `concurrent.futures.ThreadPoolExecutor` (8 workers) |
| **영상 편집** | ffmpeg (구간 추출 + concat demuxer) |
| **그리드 생성** | NumPy + Pillow |
| **병렬 처리** | asyncio + `asyncio.to_thread` (20개 동시 처리) |

### 실행

```bash
# 단일 영상 처리
python3 main/day2/video_pipeline.py 1

# 20개 영상 일괄 처리
python3 main/day2/run_all.py
```

### 결과 파일

```
main/day2/extracted_picture/{video_name}/
├── grid_00000s.jpg   # 0~9s 그리드
├── grid_00010s.jpg   # 10~19s 그리드
├── ...
└── condensed.mp4     # 관련 구간 압축 영상
```

---

## 의존성

```bash
pip install numpy pandas scipy scikit-learn matplotlib \
            torch numba opencv-python pillow \
            google-generativeai google-genai
```

ffmpeg는 시스템에 별도 설치가 필요합니다.

```bash
apt-get install ffmpeg
```

---

## API 키 설정

```json
// secret/api_key.json
{
  "gemini": "YOUR_GEMINI_API_KEY"
}
```
