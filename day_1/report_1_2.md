# 문제 1-2: 6비트 정수 곱셈기 — 접근법 및 결과

## 1. 아키텍처 선택: Causal Transformer (ConceptMultiplier)

### 선택한 구조
```
ConceptMultiplier(d_model=32, n_heads=4, n_layers=3, d_ff=128)
Trainable Parameters (P₂): 39,010
```

| 구성 요소 | 상세 |
|---|---|
| 입력 표현 | 2-token vocabulary Embedding (0/1) + Learned Positional Encoding |
| 시퀀스 형식 | `[A₀..A₅, B₀..B₅, P₀..P₁₁]` (LSB-First, 총 24토큰) |
| 인코더 | TransformerEncoder, 3 layers, norm_first=True, GELU |
| 어텐션 | 4 heads, causal mask (autoregressive) |
| 출력 | Linear head (weight-tied with embedding) → 2-class (0 or 1) |
| 손실 | CrossEntropyLoss (12개 출력 비트 위치에만 적용) |

### 왜 이 아키텍처인가?
- **Causal (자기회귀) Transformer**: 곱셈은 A, B의 모든 비트 조합에 걸친 전달 올림(carry) 계산이 필요합니다. Causal mask를 적용하면 각 출력 비트 Pᵢ를 예측할 때 앞 비트들에만 attend 하며 순차적 carry propagation을 모방할 수 있습니다.
- **Learned Positional Embedding**: 고정(Sinusoidal) 대신 학습형으로, 모델이 특정 비트 위치(Aᵢ, Bⱼ)를 자유롭게 매칭·라우팅할 수 있도록 합니다.
- **Weight Tying**: 임베딩과 출력 헤드 가중치를 공유하여 파라미터 수를 최소화합니다.
- **AdamW + Cosine Annealing**: weight_decay=0.01로 일반화를 보조하고, cosine lr schedule로 수렴 안정성을 확보합니다.

---

## 2. 학습 곡선 (Loss & Accuracy vs. Epoch)

> 메인 모델: `d_model=32, n_heads=4, n_layers=3, d_ff=128` (39,010 params)
> 학습 데이터: 100,000샘플 / 검증: 10,000샘플 / batch=256

| Epoch | Train Loss | Val Acc |
|------:|----------:|--------:|
|    10 |    0.2451 |  29.64% |
|    20 |    0.1392 |  62.76% |
|    30 |    0.0840 |  87.90% |
|    40 |    0.0509 |  95.80% |
|    50 |    0.0370 |  92.21% |
|    60 |    0.0298 |  97.98% |
|    70 |    0.0251 |  95.44% |
| **80** | **0.0211** | **99.05% ✓** |

```
Val Acc (%)
100 |                                               ●  ← 99.05% 달성
 98 |                                         ●
 96 |                              ●
 92 |                   ●                  ●
 88 |         ●
 63 |   ●
 30 | ●
    +----+----+----+----+----+----+----+----→ Epoch
    0   10   20   30   40   50   60   70   80
```

**관찰:**
- 1~30에폭: 급격한 학습 (loss 0.24 → 0.08, acc 30% → 88%)
- 40~70에폭: loss는 계속 하강하나 accuracy가 일시 등락 (95%↔98%) — carry-bit 일반화 불안정 구간
- 80에폭: **99.05% 달성 → 조기 종료**

---

## 3. 모델 크기 vs. 정확도

> 각 config를 50 에폭 학습 후 Val Accuracy 측정 (실측값)

| 레이블 | d_model | n_layers | d_ff | 파라미터 수 | Acc@50ep |
|---|---|---|---|---:|---:|
| tiny | 8 | 2 | 16 | 1,426 | 10.78% |
| small-a | 8 | 2 | 32 | 1,970 | 7.10% |
| small-b | 16 | 2 | 64 | 7,010 | 15.35% |
| medium | 16 | 3 | 64 | 10,290 | 46.63% |
| medium-L | 24 | 3 | 96 | 22,346 | 81.72% |
| **original** | **32** | **3** | **128** | **39,010** | **86.63%** (→99.05%@80ep) |

```
Acc@50ep (%)
 90 |                                               ●  original
 82 |                                    ●  medium-L
 47 |                         ●  medium
 15 |              ●  small-b
 11 | ●  tiny
  7 |    ●  small-a
    +----+------+------+-------+-------+--------→ Params
    0   1K     2K    7K     10K     22K    39K
```

**트렌드:**
- 파라미터 < 10K: carry propagation 표현 capacity 부족, 50에폭 내 50% 미달
- 10K → 22K: 급격한 성능 향상 (47% → 82%), d_model 증가의 핵심 구간
- 22K → 39K: 추가 향상 (82% → 87%@50ep, 99%@80ep)
- **d_model 증가(attention capacity ∝ d_model²)** 가 n_layers 증가보다 효과적

---

## 요약

| 항목 | 값 |
|---|---|
| 아키텍처 | Causal Transformer Encoder (ConceptMultiplier) |
| 파라미터 수 (P₂) | **39,010** |
| 99% 달성 에폭 | **80 / 200** |
| 최종 Val Accuracy | **99.05%** |
| 학습 시간 | ~6분 (GPU) |
