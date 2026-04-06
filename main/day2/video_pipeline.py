"""
video_pipeline.py

2-Agent 비디오 분석 파이프라인:

  [TimestampAgent]
    1. 영상을 5초 간격 10x10 그리드로 스캔
    2. Gemini → 관련 시점(초) 리스트 추출
    3. 인접 시점 병합 → 구간(segments) 생성
    4. ffmpeg로 구간만 잘라서 하나의 압축 영상으로 병합

  [AnswerAgent]
    1. 압축 영상을 Gemini File API로 업로드
    2. CoT 5단계 프롬프트로 최종 답변 추론
"""
import os
import re
import cv2
import json
import time
import subprocess
import tempfile
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ──────────────────────────────────────────────────────────
# API 초기화
# ──────────────────────────────────────────────────────────
KEY_PATH = "/workspace/hackathon/secret/api_key.json"
try:
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        _key = json.load(f).get("gemini")
        if not _key:
            raise ValueError("'gemini' 키 없음")
        client = genai.Client(api_key=_key)
except Exception as _e:
    print(f"[오류] API 키 로드 실패: {_e}")
    raise

MODEL_NAME       = "gemini-3-flash-preview"
BASE_EXTRACT_DIR = "/workspace/hackathon/main_2/extracted_picture"


# ══════════════════════════════════════════════════════════
# TimestampAgent — 관련 시점 추출 + 압축 영상 생성
# ══════════════════════════════════════════════════════════
class TimestampAgent:
    """
    역할:
      - 영상 전체를 그리드로 스캔 → Gemini로 관련 시점 식별
      - 관련 구간만 잘라 하나의 압축 영상(condensed video)으로 병합
    """
    GRID_INTERVAL  = 1       # 그리드 셀 간격 (초)
    GRID_COLS      = 10      # 그리드 열 수
    CELL_SIZE      = (384, 216)
    MERGE_GAP      = 2       # 병합 gap (초)
    BUFFER_SEC     = 0       # 구간 앞뒤 확장 (초)

    def __init__(self, video_path: str, save_dir: str):
        self.video_path = video_path
        self.save_dir   = save_dir
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        self.fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps)
        cap.release()

    # ── 프레임 추출 ───────────────────────────────────────

    def _extract_frame(self, sec: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.CELL_SIZE)

        # 타임스탬프 자막 — 반투명 검정 배경 + 흰색 텍스트
        label      = f"{sec}s"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3
        x, y = 4, 20
        # 반투명 배경 오버레이
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        # 흰색 텍스트
        cv2.putText(img, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img

    GRID_ROWS      = 1       # 그리드 행 수 → 그리드 1장 = 10×1 = 10초

    def _build_grids(self) -> list[tuple[Image.Image, int, int]]:
        """1초 간격 전체 스캔 → 10셀(10×1)씩 그리드 생성"""
        seconds        = list(range(0, self.total_sec + 1, self.GRID_INTERVAL))
        cw, ch         = self.CELL_SIZE
        cells_per_grid = self.GRID_COLS * self.GRID_ROWS  # 10

        # 병렬 프레임 추출
        raw: dict[int, Optional[np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(self._extract_frame, s): s for s in seconds}
            for f in as_completed(futs):
                s = futs[f]
                raw[s] = f.result()

        grids: list[tuple[Image.Image, int, int]] = []
        for chunk_start in range(0, len(seconds), cells_per_grid):
            chunk = seconds[chunk_start: chunk_start + cells_per_grid]
            cells = []
            for s in chunk:
                cell = raw.get(s)
                if cell is None:
                    cell = np.zeros((ch, cw, 3), dtype=np.uint8)
                cells.append(cell)

            # 25열 기준 패딩
            while len(cells) % self.GRID_COLS:
                cells.append(np.zeros((ch, cw, 3), dtype=np.uint8))

            rows  = [np.hstack(cells[i:i+self.GRID_COLS])
                     for i in range(0, len(cells), self.GRID_COLS)]
            grid  = Image.fromarray(np.vstack(rows))
            start = chunk[0]
            end   = chunk[-1]

            path  = os.path.join(self.save_dir, f"grid_{start:05d}s.jpg")
            grid.save(path, format="JPEG", quality=85)
            print(f"  [TimestampAgent] 그리드 저장: {path}  ({start}s~{end}s)")
            grids.append((grid, start, end))

        return grids

    # ── Gemini: 관련 시점 추출 ────────────────────────────

    def _query_timestamps(
        self, grids: list[Image.Image], query: str
    ) -> list[int]:
        grid_imgs = [g for g, *_ in grids]
        total     = self.total_sec

        prompt = f"""These images are 1-second interval video frame grids covering 0s ~ {total}s.
Each frame is labeled in RED with its timestamp.

Task: Find ALL timestamps that are relevant to the following query.

Query: "{query}"

Rules:
- Include every timestamp where the query subject is visible or occurring
- Include timestamps just before and after relevant moments
- Do NOT filter by scene transitions — include all relevant seconds

Return ONLY a JSON integer array of relevant timestamps.
Example: [8, 9, 54, 55, 134, 135]
Return [] if nothing relevant found. No other text."""

        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=grid_imgs + [prompt]
        )
        raw = res.text.strip()
        print(f"  [TimestampAgent] Gemini 응답: {raw[:200]}")

        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            try:
                return [int(x) for x in json.loads(m.group(0))
                        if isinstance(x, (int, float))]
            except Exception:
                pass
        return [int(x) for x in re.findall(r'\b\d+\b', raw)]

    # ── 시점 → 구간 병합 ─────────────────────────────────

    def _merge_to_segments(
        self, seconds: list[int]
    ) -> list[tuple[int, int]]:
        """인접 시점을 하나의 구간으로 병합하고 앞뒤 버퍼 추가"""
        if not seconds:
            return []
        secs = sorted(set(seconds))
        segments, start, end = [], secs[0], secs[0]

        for s in secs[1:]:
            if s - end <= self.MERGE_GAP:
                end = s
            else:
                segments.append((start, end))
                start = end = s
        segments.append((start, end))

        # 버퍼 확장 + 영상 범위 클램프
        buffered = [
            (max(0, s - self.BUFFER_SEC),
             min(self.total_sec, e + self.BUFFER_SEC))
            for s, e in segments
        ]
        return buffered

    # ── ffmpeg: 구간 잘라 병합 ────────────────────────────

    def _build_condensed_video(
        self, segments: list[tuple[int, int]], output_path: str
    ) -> str:
        """
        ffmpeg로 각 구간을 추출한 뒤 하나의 영상으로 이어 붙인다.
        구간 시작마다 타임스탬프 자막(drawtext)을 오버레이한다.
        """
        if not segments:
            raise ValueError("병합할 구간이 없습니다.")

        tmp_dir   = tempfile.mkdtemp()
        clip_paths: list[str] = []

        for i, (s, e) in enumerate(segments):
            duration = max(1, e - s)
            clip_out = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")

            # 구간 추출 + 좌상단 타임스탬프 자막
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(s),
                "-i", self.video_path,
                "-t",  str(duration),
                "-vf", (
                    f"drawtext=text='Segment {i+1}\\: {s}s~{e}s':"
                    f"fontcolor=red:fontsize=24:x=10:y=10"
                ),
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                clip_out
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                print(f"  [경고] 구간 {i+1} 추출 실패: {result.stderr[-200:]}")
                continue
            clip_paths.append(clip_out)
            print(f"  [TimestampAgent] 클립 {i+1}: {s}s~{e}s ({duration}s) → {clip_out}")

        if not clip_paths:
            raise RuntimeError("추출된 클립이 없습니다.")

        # concat demuxer용 리스트 파일 생성
        list_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        # 병합
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(
            concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"영상 병합 실패: {result.stderr.decode()[-300:]}"
            )

        total_dur = sum(e - s for s, e in segments)
        print(f"  [TimestampAgent] 압축 영상 생성 완료: {output_path}")
        print(f"    원본 {self.total_sec}s → 압축 {total_dur}s "
              f"({total_dur/max(1,self.total_sec)*100:.1f}%)")
        return output_path

    # ── 공개 API ─────────────────────────────────────────

    def run(self, query: str) -> tuple[str, list[tuple[int, int]]]:
        """
        Returns:
          condensed_video_path : 압축 영상 경로
          segments             : [(start_sec, end_sec), ...]
        """
        print(f"\n[TimestampAgent] 쿼리: {query}")
        print(f"  영상 길이: {self.total_sec}s")

        # 1. 그리드 생성
        grids = self._build_grids()

        # 2. Gemini로 관련 시점 추출
        raw_secs = self._query_timestamps(grids, query)
        print(f"  관련 시점: {sorted(raw_secs)}")

        if not raw_secs:
            print("  [경고] 관련 시점 없음 → 전체 영상 사용")
            segments = [(0, self.total_sec)]
        else:
            segments = self._merge_to_segments(raw_secs)
        print(f"  병합 구간: {segments}")

        # 3. 압축 영상 생성
        output_path = os.path.join(self.save_dir, "condensed.mp4")
        self._build_condensed_video(segments, output_path)

        return output_path, segments


# ══════════════════════════════════════════════════════════
# AnswerAgent — 압축 영상 기반 CoT 최종 추론
# ══════════════════════════════════════════════════════════
class AnswerAgent:
    """
    역할:
      - 압축 영상을 Gemini File API로 업로드
      - CoT 5단계 프롬프트로 최종 정답 추론
    """
    THINKING_BUDGET = 8192

    def _upload_video(self, video_path: str):
        """Gemini File API에 영상 업로드 후 처리 완료 대기"""
        print(f"  [AnswerAgent] 영상 업로드 중: {video_path}")
        video_file = client.files.upload(file=video_path)

        # 처리 완료까지 폴링
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Gemini 파일 처리 실패: {video_file.state}")

        print(f"  [AnswerAgent] 업로드 완료: {video_file.uri}")
        return video_file

    # ── Task 감지 ─────────────────────────────────────────

    @staticmethod
    def _detect_task(question: str) -> str:
        """
        질문 유형 분류:
          sequence   — N번째 항목 찾기 (5th, first, 번째)
          count      — 총 개수 세기 (how many, distinct)
          appearance — 특정 사물/브랜드/색상 식별
        """
        lower = question.lower()
        if any(kw in lower for kw in [
            "fifth","5th","first","second","third","fourth","sixth",
            "7th","8th","9th","10th","번째","nth","which instrument",
            "순서","in a row","in sequence",
        ]):
            return "sequence"
        if any(kw in lower for kw in [
            "how many","distinct","count","total number","몇","개수",
            "number of","how often","times does",
        ]):
            return "count"
        return "appearance"

    # ── Task별 CoT 프롬프트 ───────────────────────────────

    def _build_prompt(
        self,
        question: str,
        seg_desc: str,
        n_segs: int,
        ctx_block: str,
        task: str,
    ) -> str:
        header = (
            f"You are watching a condensed video built from the most relevant segments.\n"
            f"Segments: {seg_desc}  (total {n_segs} clip(s))\n"
            f"Each clip starts with a RED caption showing its original timestamp.\n"
            f"{ctx_block}"
        )

        if task == "sequence":
            return f"""{header}
This is a SEQUENCE question — you must find the Nth occurrence of a specific subject/event.

**STEP 1 — BUILD THE OCCURRENCE LIST**
Watch the entire video. Each time the target subject appears in a NEW close-up shot,
add one line to a numbered list:
  #1 (Xs): <subject name / description> — actively playing / performing? YES or NO
  #2 (Xs): ...
  (Only count shots where the subject is ACTIVELY engaged, as the question specifies.)

**STEP 2 — FILTER & RENUMBER**
Remove any entries marked NO from STEP 1.
Renumber the remaining entries 1, 2, 3, ... in order.

**STEP 3 — IDENTIFY THE TARGET**
The question asks for the Nth entry. State clearly:
  "After filtering, entry #N is: <subject> at Xs."

**STEP 4 — QUESTION**
{question}

**STEP 5 — FINAL ANSWER**
Match the subject from STEP 3 to the correct option letter.
Write on its own line:
ANSWER: <single uppercase letter>"""

        if task == "count":
            return f"""{header}
This is a COUNT question — you must count the total number of distinct items/scenes.

**STEP 1 — LIST ALL DISTINCT ITEMS**
Watch the entire video. Each time you see a NEW distinct item (scene, angle, subject),
add it to a list:
  Item 1 (Xs): <description>
  Item 2 (Xs): <description>
  ...
If the video returns to a previously seen item, note it as "REPEAT of Item N" — do NOT add a new entry.

**STEP 2 — DEDUPLICATE**
Review your list and remove any duplicates or repeats.
Count only the unique items.

**STEP 3 — VERIFY THE COUNT**
State: "Total unique items = N."
Double-check by reviewing the list one more time.

**STEP 4 — QUESTION**
{question}

**STEP 5 — FINAL ANSWER**
Match the count N to the correct option letter.
Write on its own line:
ANSWER: <single uppercase letter>"""

        # appearance (default)
        return f"""{header}
This is an APPEARANCE question — you must identify a specific visual detail.

**STEP 1 — FIND THE RELEVANT MOMENT**
Watch the video and identify the exact moment(s) where the answer is visible.
For each candidate moment write:
  (Xs): <what is visible — brand, label, color, text, person, object>

**STEP 2 — ANALYZE VISUAL DETAILS**
For the best candidate from STEP 1:
  - What text, logo, or label is visible?
  - What color, shape, or distinctive feature can you see?
  - How confident are you? (high / medium / low)

**STEP 3 — MATCH TO OPTIONS**
Compare your observation from STEP 2 against the answer options.
State: "The visible detail is ____, which matches option X."

**STEP 4 — QUESTION**
{question}

**STEP 5 — FINAL ANSWER**
Write on its own line:
ANSWER: <single uppercase letter>"""

    # ── 공개 API ─────────────────────────────────────────

    def run(
        self,
        condensed_video_path: str,
        question: str,
        segments: list[tuple[int, int]],
        pipeline_context: str = "",
    ) -> str:
        """
        Task 유형 감지 → 맞춤 CoT 프롬프트 → 최종 정답 (A-Z)
        """
        task = self._detect_task(question)
        print(f"\n[AnswerAgent] Task 유형: {task.upper()}")

        video_file = self._upload_video(condensed_video_path)

        seg_desc  = ", ".join(f"{s}s~{e}s" for s, e in segments)
        ctx_block = (f"\n[Pipeline Context]\n{pipeline_context}\n"
                     if pipeline_context else "")

        prompt = self._build_prompt(question, seg_desc, len(segments), ctx_block, task)

        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.THINKING_BUDGET
                )
            )
        )
        raw = res.text.strip()
        print(f"  [AnswerAgent/{task}]\n{'·'*55}\n{raw[:1200]}\n{'·'*55}")

        # 파일 삭제
        try:
            client.files.delete(name=video_file.name)
        except Exception:
            pass

        # ANSWER: X 파싱
        m = re.search(r'ANSWER\s*[:\-]\s*([A-Z])', raw)
        if m:
            return m.group(1)
        matches = re.findall(r'\b([A-Z])\b', raw)
        return matches[-1] if matches else "A"


# ══════════════════════════════════════════════════════════
# VideoPipeline — 전체 흐름 제어
# ══════════════════════════════════════════════════════════
class VideoPipeline:
    def __init__(self, video_path: str):
        video_name    = os.path.splitext(os.path.basename(video_path))[0]
        self.save_dir = os.path.join(BASE_EXTRACT_DIR, video_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.video_path      = video_path
        self.timestamp_agent = TimestampAgent(video_path, self.save_dir)
        self.answer_agent    = AnswerAgent()

    @staticmethod
    def _extract_scan_query(prompt: str) -> str:
        """
        routing_json에 CLIP 쿼리가 없을 때,
        Gemini로 프롬프트에서 시각 검색용 핵심 키워드를 추출한다.
        """
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"""Extract a short English visual search query (5-10 words) from this question.
Focus on what to LOOK FOR visually in the video frames.
Output ONLY the search query, nothing else.

Question: {prompt[:500]}"""
        )
        query = res.text.strip().strip('"').strip("'")
        print(f"  [자동 스캔 쿼리 생성] {query}")
        return query

    def run(self, routing_json: dict, original_prompt: str = "") -> str:
        # 쿼리 추출
        scan_query = (routing_json
                      .get("anchor_event", {})
                      .get("generated_queries", {})
                      .get("for_CLIP", ""))
        final_q = routing_json.get("target_event", {}).get(
            "generated_query_for_VLM", ""
        )
        if original_prompt:
            final_q = original_prompt

        # scan_query가 없으면 프롬프트에서 자동 추출
        if not scan_query:
            scan_query = self._extract_scan_query(final_q)

        print(f"\n{'='*55}")
        print(f"  스캔 쿼리 : {scan_query}")
        print(f"  최종 질문 : {final_q[:80]}")
        print(f"{'='*55}")

        t0 = time.time()

        # ── TimestampAgent ───────────────────────────────
        condensed_path, segments = self.timestamp_agent.run(scan_query)

        pipeline_ctx = (
            f"- Original video: {self.timestamp_agent.total_sec}s\n"
            f"- Relevant segments: {', '.join(f'{s}s~{e}s' for s,e in segments)}\n"
            f"- Search query: \"{scan_query}\""
        )

        # ── AnswerAgent ──────────────────────────────────
        answer = self.answer_agent.run(
            condensed_path, final_q, segments, pipeline_ctx
        )

        print(f"\n  [Pipeline] 총 소요: {time.time()-t0:.1f}s")
        print(f"  [Pipeline] 최종 정답: {answer}")
        return answer


# ══════════════════════════════════════════════════════════
# 외부 진입점
# ══════════════════════════════════════════════════════════
def process_task(
    video_path: str,
    routing_json: dict,
    original_prompt: str = "",
) -> str:
    return VideoPipeline(video_path).run(routing_json, original_prompt)


# ── 단독 테스트 ──────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from prompt_analysis import analyze_prompt

    SAMPLE_DIR = "/workspace/hackathon/main_2/sample_test"

    arg = sys.argv[1] if len(sys.argv) > 1 else "1"

    if arg in ("1", "2", "3"):
        _vp = f"{SAMPLE_DIR}/video{arg}.mp4"
        _pp = open(f"{SAMPLE_DIR}/prompt{arg}.txt").read().strip()
    else:
        _vp = arg
        _pp = open(sys.argv[2]).read().strip() if len(sys.argv) > 2 else ""

    print(f"영상: {_vp}")
    print(f"프롬프트: {_pp[:80]}...")

    # prompt_analysis로 routing_json 생성 → for_CLIP 스캔 쿼리 포함
    print("\n[프롬프트 분석 중...]")
    _routing = analyze_prompt(_pp)
    print(f"  scan query (for_CLIP): {_routing.get('anchor_event', {}).get('generated_queries', {}).get('for_CLIP', '(없음)')}")

    print(process_task(_vp, _routing, _pp))
