"""
run_all.py

/workspace/hackathon_data 의 video1~20.mp4 와 prompt1~20.txt 를
video_pipeline.py 를 사용해 20개 동시 비동기 처리 후
1~20 순서대로 answer 를 출력/저장합니다.
"""
import os
import sys
import asyncio
import time

# ── 경로 설정 ──────────────────────────────────────────────
DATA_DIR   = "/workspace/hackathon_data"
VIDEO_DIR  = os.path.join(DATA_DIR, "videos")
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
RESULT_FILE = "/workspace/hackathon/main_2/answer.txt"

sys.path.insert(0, "/workspace/hackathon/main_2")
from prompt_analysis import analyze_prompt
from video_pipeline import process_task


# ══════════════════════════════════════════════════════════
# 단일 비디오 처리 (비동기)
# ══════════════════════════════════════════════════════════
async def process_one(video_id: int, semaphore: asyncio.Semaphore) -> tuple[int, str]:
    video_path  = os.path.join(VIDEO_DIR,  f"video{video_id}.mp4")
    prompt_path = os.path.join(PROMPT_DIR, f"prompt{video_id}.txt")

    prompt_text = open(prompt_path, encoding="utf-8").read().strip()

    async with semaphore:
        print(f"\n[{video_id:02d}] ▶ 시작")
        t0 = time.time()
        try:
            # 1. 프롬프트 분석 → routing_json
            routing_json = await asyncio.to_thread(analyze_prompt, prompt_text)

            # 2. video_pipeline 실행
            answer = await asyncio.to_thread(
                process_task, video_path, routing_json, prompt_text
            )
            print(f"[{video_id:02d}] ✅ 정답: {answer}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[{video_id:02d}] ❌ 에러: {e}")
            answer = "A"

    return video_id, answer


# ══════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════
async def main():
    # 20개 동시 처리 (세마포어로 동시성 제한 없음 — 필요시 조정)
    semaphore = asyncio.Semaphore(20)

    t_start = time.time()
    print("=" * 60)
    print("  hackathon video pipeline — 20개 동시 처리 시작")
    print("=" * 60)

    tasks = [process_one(i, semaphore) for i in range(1, 21)]
    results = await asyncio.gather(*tasks)

    # video_id 순서대로 정렬
    answer_dict = dict(results)
    ordered_answers = [answer_dict.get(i, "A") for i in range(1, 21)]
    submission = "".join(ordered_answers)

    print("\n" + "=" * 60)
    print("  처리 완료!")
    print(f"  총 소요: {time.time()-t_start:.1f}s")
    print()
    for i, ans in enumerate(ordered_answers, 1):
        print(f"  [{i:02d}] {ans}")
    print()
    print(f"  제출 문자열: {submission}")
    print("=" * 60)

    # 파일 저장
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("# 1~20번 비디오 정답\n")
        for i, ans in enumerate(ordered_answers, 1):
            f.write(f"{i}: {ans}\n")
        f.write(f"\n제출 문자열: {submission}\n")

    print(f"\n결과 저장: {RESULT_FILE}")
    return submission


if __name__ == "__main__":
    asyncio.run(main())
