# Video Analysis Pipeline 보고서

## 1. 개요

`video_pipeline.py`는 영상과 질문(prompt)을 입력받아 자동으로 정답(A-Z)을 추론하는 2-Agent 비디오 분석 파이프라인입니다.

---

## 2. 전체 구조

```
입력: video.mp4 + prompt 텍스트
            │
            ▼
    [VideoPipeline.run()]
            │
            ├─── 스캔 쿼리 자동 추출 (Gemini)
            │         프롬프트에서 시각 검색용 핵심 키워드 생성
            │
            ▼
    ┌─────────────────────┐
    │   TimestampAgent    │
    │                     │
    │  1. 1초 간격 프레임  │
    │     전수 추출        │
    │  2. 10×1 그리드 생성 │
    │  3. Gemini →        │
    │     관련 시점 추출   │
    │  4. ffmpeg →        │
    │     압축 영상 생성   │
    └──────────┬──────────┘
               │ condensed.mp4
               ▼
    ┌─────────────────────┐
    │    AnswerAgent      │
    │                     │
    │  1. Task 유형 감지   │
    │  2. 영상 업로드      │
    │     (File API)      │
    │  3. Task별 CoT      │
    │     프롬프트 추론    │
    │  4. ANSWER: X 파싱  │
    └──────────┬──────────┘
               │
               ▼
          최종 정답 (A-Z)
```

---

## 3. 컴포넌트 상세

### 3.1 VideoPipeline

전체 흐름을 제어하는 오케스트레이터입니다.

**주요 역할**
- `routing_json`에서 `for_CLIP` 스캔 쿼리 추출
- 쿼리가 없을 경우 Gemini를 통해 프롬프트에서 자동 생성
- TimestampAgent → AnswerAgent 순서로 실행

**스캔 쿼리 자동 추출**
```
프롬프트 전체 텍스트
        │
        ▼  Gemini (5~10단어 시각 검색 키워드 추출)
        ▼
"Close-up of instruments being played, Bolero performance"
```

---

### 3.2 TimestampAgent

영상 전체를 그리드 이미지로 변환하여 관련 시점을 식별하고, 해당 구간만 잘라 압축 영상을 생성합니다.

#### 설정값

| 항목 | 값 | 설명 |
|---|---|---|
| `GRID_INTERVAL` | 1초 | 모든 프레임 전수조사 |
| `GRID_COLS × GRID_ROWS` | 10 × 1 | 그리드 1장 = 10초 분량 |
| `CELL_SIZE` | 384 × 216 px | 고화질 셀 |
| `MERGE_GAP` | 2초 | 2초 이내 시점 → 하나의 클립으로 병합 |
| `BUFFER_SEC` | 0초 | 관련 시점만 정확히 추출, 패딩 없음 |

#### 처리 흐름

```
Step 1. 프레임 추출
  - ThreadPoolExecutor (max_workers=8) 병렬 추출
  - 각 셀: 384×216 리사이즈
  - 타임스탬프 오버레이 (반투명 검정 배경 + 흰색 텍스트)

Step 2. 그리드 생성
  - 10초 단위로 10×1 그리드 이미지 생성
  - extracted_picture/{video_name}/grid_XXXXXS.jpg 저장

Step 3. Gemini 시점 추출
  - 모든 그리드 이미지 + 쿼리 → Gemini
  - 관련 시점을 JSON 정수 배열로 반환
  - re.search(r'\[.*?\]') 로 파싱

Step 4. 구간 병합 및 압축 영상 생성
  - MERGE_GAP(2초) 이내 시점 → 하나의 구간으로 병합
  - ffmpeg로 각 구간 추출 + 타임스탬프 자막(drawtext) 오버레이
  - concat demuxer로 단일 condensed.mp4 생성
```

#### Gemini 쿼리 프롬프트 (TimestampAgent)

```
These images are 1-second interval video frame grids covering 0s ~ {total}s.
Each frame is labeled with its timestamp.

Task: Find ALL timestamps that are relevant to the following query.
Query: "{query}"

Rules:
- Include every timestamp where the query subject is visible or occurring
- Include timestamps just before and after relevant moments

Return ONLY a JSON integer array.
```

---

### 3.3 AnswerAgent

압축 영상을 Gemini File API로 업로드하고, Task 유형에 따른 CoT 프롬프트로 최종 답변을 추론합니다.

#### Task 자동 감지

| Task | 감지 키워드 | 예시 질문 |
|---|---|---|
| **sequence** | fifth, 5th, first, 번째, which instrument, 순서 | "5번째 등장하는 악기는?" |
| **count** | how many, distinct, count, 몇, number of | "총 몇 개의 장면이 있나요?" |
| **appearance** | (나머지 전부) | "어떤 브랜드의 드릴을 사용했나요?" |

#### Task별 CoT 전략

**sequence (순서 찾기)**
```
STEP 1: 전체 영상에서 대상 등장 목록 작성
        #1 (Xs): <대상> — 활성 여부 YES/NO
        #2 (Xs): ...

STEP 2: NO 항목 제거 후 재번호 부여
        → 1, 2, 3, ...

STEP 3: N번째 항목 특정
        "entry #N is: <대상> at Xs"

STEP 4: 질문 확인

STEP 5: ANSWER: X
```

**count (개수 세기)**
```
STEP 1: 고유 항목 목록 작성
        Item 1 (Xs): <설명>
        Item 2 (Xs): <설명>
        재등장 시 "REPEAT of Item N" 표시

STEP 2: 중복 제거 후 고유 개수 산출

STEP 3: "Total unique items = N" 검증

STEP 4: 질문 확인

STEP 5: ANSWER: X
```

**appearance (시각 식별)**
```
STEP 1: 답변이 보이는 시점 탐색
        (Xs): <브랜드/색상/텍스트/인물>

STEP 2: 시각적 디테일 분석
        - 텍스트, 로고, 색상, 형태
        - 신뢰도 (high/medium/low)

STEP 3: 옵션과 대조
        "The visible detail is ____, which matches option X"

STEP 4: 질문 확인

STEP 5: ANSWER: X
```

#### 답변 파싱

```python
# 1순위: "ANSWER: X" 또는 "ANSWER - X" 패턴
m = re.search(r'ANSWER\s*[:\-]\s*([A-Z])', raw)

# 2순위 (fallback): 마지막 단독 대문자
matches = re.findall(r'\b([A-Z])\b', raw)
return matches[-1] if matches else "A"
```

---

## 4. 기술 스택

| 항목 | 내용 |
|---|---|
| **모델** | `gemini-3-flash-preview` |
| **SDK** | `google.genai` (신규 패키지) |
| **Reasoning** | `ThinkingConfig(thinking_budget=8192)` — AnswerAgent 전용 |
| **프레임 추출** | OpenCV + `concurrent.futures.ThreadPoolExecutor` |
| **영상 편집** | ffmpeg (구간 추출 + concat demuxer) |
| **그리드 생성** | NumPy + Pillow |
| **파일 업로드** | `client.files.upload()` + PROCESSING 상태 폴링 |

---

## 5. 실행 방법

```bash
cd /workspace/hackathon/main_2

# 번호로 실행 (sample_test/ 자동 매핑)
python3 video_pipeline.py 1
python3 video_pipeline.py 2
python3 video_pipeline.py 3

# 직접 경로 지정
python3 video_pipeline.py /path/to/video.mp4 /path/to/prompt.txt
```

---

## 6. 출력 파일 구조

```
extracted_picture/
└── {video_name}/
    ├── grid_00000s.jpg     # 0~9s 그리드
    ├── grid_00010s.jpg     # 10~19s 그리드
    ├── ...
    └── condensed.mp4       # 관련 구간 압축 영상
```

---

## 7. 알려진 이슈

| 이슈 | 원인 | 비고 |
|---|---|---|
| 장편 영상(~800s)에서 모든 시점 반환 | 스캔 쿼리가 영상 전체에 해당하는 경우 압축률 0% | 쿼리 구체화 필요 |
| Gemini File API ACTIVE 대기 | 업로드 후 처리 완료 전 요청 시 오류 | 폴링 로직으로 처리 중 |
