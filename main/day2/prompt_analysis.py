import os
import json
import google.generativeai as genai

# 1. API 키 파일 경로 설정 및 읽기
KEY_PATH = "/workspace/hackathon/secret/api_key.json"

try:
    with open(KEY_PATH, 'r', encoding='utf-8') as f:
        secret_data = json.load(f)
        api_key = secret_data.get("gemini")
        
        if not api_key:
            raise ValueError("JSON 파일 내에 'gemini' 키가 없습니다.")
            
        # 2. 불러온 키로 Gemini API 설정
        genai.configure(api_key=api_key)
        
except FileNotFoundError:
    print(f"[오류] API 키 파일을 찾을 수 없습니다: {KEY_PATH}")
    exit(1)
except json.JSONDecodeError:
    print("[오류] API 키 파일의 JSON 형식이 올바르지 않습니다.")
    exit(1)
except Exception as e:
    print(f"[오류] API 키 로드 중 문제 발생: {e}")
    exit(1)

# --- 이하 기존 코드 동일 ---

SYSTEM_INSTRUCTION = """
You are the 'Task Commander' for an advanced Video Analysis Agent system.
Your primary role is to analyze the user's video-related question, decompose the logical steps, and generate specific sub-queries for various AI tools.

[Instructions]
1. Temporal Relationships: Check if the prompt has a temporal dependency. Convert offsets to SECONDS (e.g., 7 minutes before = -420).
2. Generate Tool-Specific Queries: Generate English keywords for CLAP (audio), CLIP (vision), or OCR (text).
3. Format Options: Parse the multiple-choice options correctly.

You MUST output ONLY a valid JSON object matching this schema:
{
  "task_category": "Easy_Appearance" | "Medium_Spatial" | "Hard_Temporal_Anchor",
  "temporal_offset_seconds": 0,
  "anchor_event": {
    "is_exist": true,
    "original_context": "string",
    "generated_queries": { "for_CLAP": "string", "for_CLIP": "string", "for_OCR": ["string"] }
  },
  "target_event": {
    "generated_query_for_VLM": "string",
    "options": ["A) full option text", "B) full option text", "C) full option text"]
  }
}

IMPORTANT for options field: Always include the FULL option text (e.g. "A) DeWALT", "B) Makita"), NOT just the letter.
If options are given as "A) DeWALT", output exactly ["A) DeWALT", "B) Makita", ...].
Never strip the description — the full text is required for visual matching.
"""

def analyze_prompt(user_prompt: str) -> dict:
    """
    가볍고 빠른 Gemini 1.5 Flash를 사용하여 텍스트 프롬프트를 JSON 지시서로 분해합니다.
    """
    model = genai.GenerativeModel(
        model_name='gemini-3.1-flash-lite-preview',
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", 
            temperature=0.1 
        )
    )
    
    try:
        response = model.generate_content(user_prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"[Error] 프롬프트 분석 실패: {e}")
        return {"task_category": "Easy_Appearance", "temporal_offset_seconds": 0}

if __name__ == "__main__":
    sample = "손흥민이 골 넣은 시점은 언제인가? (A: 이강인, B: 김민재, C: 황희찬, D: 조규성)"
    print(json.dumps(analyze_prompt(sample), indent=2, ensure_ascii=False))