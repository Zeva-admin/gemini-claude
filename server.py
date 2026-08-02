# -*- coding: utf-8 -*-
"""
Gemini Bridge Server
FastAPI сервер для общения Claude ↔ Gemini через ngrok.
Сохраняет все чаты в chats/<session_id>.jsonl
"""

import os
import json
import uuid
import io
import sys
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

# UTF-8 для Windows терминала
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────
API_KEY    = os.environ.get("GEMINI_API_KEY", "")   # задай через env или впиши сюда
MODEL_NAME = "gemini-3.5-flash-lite"                      # модель с thinking
CHATS_DIR  = "chats"
THINKING_BUDGET = 8000                               # токены на рассуждение (0 = выключить)

os.makedirs(CHATS_DIR, exist_ok=True)

app = FastAPI(title="Gemini Bridge", version="1.0.0")

# ─────────────────────────────────────────────
# ХРАНИЛИЩЕ СЕССИЙ (in-memory history)
# session_id -> list of {role, content}
# ─────────────────────────────────────────────
sessions: dict[str, list[dict]] = {}


# ─────────────────────────────────────────────
# УТИЛИТЫ
# ─────────────────────────────────────────────
def get_client() -> genai.Client:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY не задан")
    return genai.Client(api_key=API_KEY)


def build_api_contents(history: list[dict]) -> list[types.Content]:
    """Конвертирует нашу историю в формат Gemini API."""
    contents = []
    for msg in history:
        role = msg["role"]  # "user" или "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))
    return contents


def save_message(session_id: str, role: str, content: str,
                 thinking: Optional[str] = None):
    """Дозаписывает одно сообщение в JSONL файл чата."""
    path = os.path.join(CHATS_DIR, f"{session_id}.jsonl")
    record = {
        "ts": datetime.utcnow().isoformat(),
        "role": role,
        "content": content,
    }
    if thinking:
        record["thinking"] = thinking
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_thinking(full_text: str) -> tuple[str, str]:
    """
    Разделяет <think>...</think> и финальный ответ.
    Возвращает (thinking, answer).
    """
    import re
    m = re.search(r"<think>(.*?)</think>(.*)", full_text, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        answer   = m.group(2).strip()
        return thinking, answer
    # Нет блока think — всё это ответ
    return "", full_text.strip()


# ─────────────────────────────────────────────
# МОДЕЛИ ЗАПРОСОВ
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # если None — создаём новую сессию
    system_prompt: Optional[str] = None
    thinking_budget: Optional[int] = THINKING_BUDGET


class NewSessionRequest(BaseModel):
    system_prompt: Optional[str] = None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Пинг для UptimeRobot / мониторинга."""
    return {"status": "ok", "model": MODEL_NAME, "sessions": len(sessions)}


@app.get("/sessions")
def list_sessions():
    """Список активных сессий."""
    result = []
    for sid, history in sessions.items():
        result.append({
            "session_id": sid,
            "turns": len([m for m in history if m["role"] == "user"]),
        })
    return {"sessions": result}


@app.post("/session/new")
def new_session(req: NewSessionRequest = NewSessionRequest()):
    """Создаёт новую сессию, возвращает session_id."""
    sid = str(uuid.uuid4())[:8]
    sessions[sid] = []
    # Сохраняем мета-запись в файл
    path = os.path.join(CHATS_DIR, f"{sid}.jsonl")
    meta = {
        "ts": datetime.utcnow().isoformat(),
        "role": "meta",
        "content": f"Сессия создана | модель: {MODEL_NAME}",
        "system_prompt": req.system_prompt or "",
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return {"session_id": sid}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Удаляет сессию из памяти (файл остаётся)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    del sessions[session_id]
    return {"deleted": session_id}


@app.get("/session/{session_id}/history")
def get_history(session_id: str):
    """Возвращает историю сессии."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"session_id": session_id, "history": sessions[session_id]}


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Основной endpoint. Отправляет сообщение в Gemini, возвращает:
    {
      "session_id": "...",
      "thinking": "...",   # блок рассуждений (если есть)
      "response": "...",   # финальный ответ
      "model": "...",
      "turns": N
    }
    """
    # Получаем или создаём сессию
    sid = req.session_id
    if sid is None or sid not in sessions:
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = []
        # Создаём файл
        path = os.path.join(CHATS_DIR, f"{sid}.jsonl")
        meta = {
            "ts": datetime.utcnow().isoformat(),
            "role": "meta",
            "content": f"Автосессия | модель: {MODEL_NAME}",
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    history = sessions[sid]

    # Сохраняем сообщение пользователя
    history.append({"role": "user", "content": req.message})
    save_message(sid, "user", req.message)

    # Строим запрос к Gemini
    client = get_client()
    api_contents = build_api_contents(history)

    # Конфиг генерации
    thinking_cfg = None
    if req.thinking_budget and req.thinking_budget > 0:
        thinking_cfg = types.ThinkingConfig(thinking_budget=req.thinking_budget)

    gen_config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=32768,
        top_p=0.95,
        thinking_config=thinking_cfg,
    )
    if req.system_prompt:
        gen_config.system_instruction = req.system_prompt

    # Запрос к Gemini (стриминг собираем в full_text)
    try:
        full_text = ""
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=api_contents,
            config=gen_config,
        )
        for chunk in stream:
            if chunk.text:
                full_text += chunk.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")

    # Разбиваем thinking и ответ
    thinking, answer = extract_thinking(full_text)

    # Сохраняем ответ модели в историю и файл
    history.append({"role": "model", "content": full_text})
    save_message(sid, "model", answer, thinking=thinking if thinking else None)

    return {
        "session_id": sid,
        "thinking": thinking,
        "response": answer,
        "model": MODEL_NAME,
        "turns": len([m for m in history if m["role"] == "user"]),
    }


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Стриминговый endpoint — возвращает Server-Sent Events.
    Каждый chunk: data: {"token": "...", "type": "think"|"answer"|"done"}
    """
    sid = req.session_id
    if sid is None or sid not in sessions:
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = []
        path = os.path.join(CHATS_DIR, f"{sid}.jsonl")
        meta = {
            "ts": datetime.utcnow().isoformat(),
            "role": "meta",
            "content": f"Автосессия stream | модель: {MODEL_NAME}",
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    history = sessions[sid]
    history.append({"role": "user", "content": req.message})
    save_message(sid, "user", req.message)

    client = get_client()
    api_contents = build_api_contents(history)

    thinking_cfg = None
    if req.thinking_budget and req.thinking_budget > 0:
        thinking_cfg = types.ThinkingConfig(thinking_budget=req.thinking_budget)

    gen_config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=32768,
        top_p=0.95,
        thinking_config=thinking_cfg,
    )
    if req.system_prompt:
        gen_config.system_instruction = req.system_prompt

    def generate():
        import re
        full_text = ""
        in_think = False
        think_done = False

        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=api_contents,
                config=gen_config,
            )
            for chunk in stream:
                if not chunk.text:
                    continue
                token = chunk.text
                full_text += token

                # Определяем тип токена
                if "<think>" in token and not think_done:
                    in_think = True

                chunk_type = "think" if (in_think and not think_done) else "answer"

                if "</think>" in token:
                    in_think = False
                    think_done = True

                payload = json.dumps({"token": token, "type": chunk_type}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            # Сохраняем финальный результат
            thinking, answer = extract_thinking(full_text)
            history.append({"role": "model", "content": full_text})
            save_message(sid, "model", answer, thinking=thinking if thinking else None)

            # Финальное событие
            done_payload = json.dumps({
                "type": "done",
                "session_id": sid,
                "turns": len([m for m in history if m["role"] == "user"]),
            }, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            err = json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[server] Запуск на порту {port}")
    print(f"[server] Модель: {MODEL_NAME}")
    print(f"[server] Чаты сохраняются в: {CHATS_DIR}/")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")