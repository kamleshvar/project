import os
import subprocess
import gc
import shutil
import httpx
import asyncio
import time
from datetime import datetime
import urllib.parse
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- 1. ABSOLUTE PATH SPECIFICATIONS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER_PATH = os.path.join(BASE_DIR, r"llama-b9360-bin-win-cpu-x64\llama-server.exe")
MODEL_PATH = os.path.join(BASE_DIR, r"google_gemma-3-1b-it-Q4_K_M.gguf")
INDEX_HTML_PATH = os.path.join(BASE_DIR, r"index.html")
ACTIVE_DB_TRACKER = os.path.join(BASE_DIR, r"active_db.txt")
CHUNKS_CACHE_PATH = os.path.join(BASE_DIR, r"chunks_cache.json")

llama_proc = None
GLOBAL_EMB_FN = None 

def purge_ports():
    try:
        for port in ["8000", "8080"]:
            try:
                cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
                pids = subprocess.check_output(["powershell", "-Command", cmd], text=True).strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        subprocess.run(f"taskkill /F /PID {pid.strip()}", shell=True, capture_output=True)
            except: pass
        time.sleep(1)
    except: pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_proc, GLOBAL_EMB_FN
    purge_ports()
    
    print("[SYSTEM] LOADING ELITE BASE EMBEDDING ENGINE (BGE-BASE)...")
    from chromadb.utils import embedding_functions
    GLOBAL_EMB_FN = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
    
    print("[SYSTEM] INITIALIZING STABLE AI ENGINE...")
    cmd = [
        LLAMA_SERVER_PATH, "-m", MODEL_PATH, "--port", "8080", 
        "-c", "4096", "--host", "127.0.0.1", "-t", "8", "-np", "1"
    ]
    llama_proc = subprocess.Popen(cmd)
    
    server_ready = False
    for i in range(40):
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get("http://127.0.0.1:8080/health", timeout=1.0)
                if res.status_code == 200:
                    server_ready = True
                    break
        except: pass
        if (i+1) % 5 == 0: print(f"[SYSTEM] Engine warming... ({i+1}/40)")

    if server_ready: print("[SYSTEM] ELITE ARCHIVE IS LIVE.")
    yield
    if llama_proc:
        subprocess.run(f"taskkill /F /PID {llama_proc.pid}", shell=True, capture_output=True)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS_PATH = os.path.join(BASE_DIR, "sessions.json")

class InitRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    query: str
    user_name: str = "User"
    mode: str = "think"
    quoted_text: str = ""
    session_id: str = ""
    general_mode: bool = False

def load_sessions():
    if os.path.exists(SESSIONS_PATH):
        try:
            with open(SESSIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

def save_sessions(data):
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_active_db():
    if os.path.exists(ACTIVE_DB_TRACKER):
        with open(ACTIVE_DB_TRACKER, "r") as f:
            path = f.read().strip()
            if os.path.exists(path): return path
    return None

@app.get("/")
async def get_index():
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>PRESENCE ONLINE</h1>")

@app.get("/status")
async def get_status():
    return {"initialized": get_active_db() is not None}

@app.post("/clear_archive")
async def clear_archive():
    if os.path.exists(ACTIVE_DB_TRACKER): os.remove(ACTIVE_DB_TRACKER)
    return {"status": "success"}

@app.post("/initialize")
async def initialize(req: InitRequest):
    try:
        import chromadb
        new_vault = os.path.join(BASE_DIR, f"vault_{int(time.time())}")
        
        # Enhanced Chunking: Split by double newline, then by single newline if needed, 
        # but also cap chunk sizes for better embedding quality
        raw_chunks = [p.strip() for p in req.text.split("\n\n") if p.strip()]
        if len(raw_chunks) < 2:
            raw_chunks = [p.strip() for p in req.text.split("\n") if p.strip()]
        
        # Further refine chunks: ensure they aren't massive
        chunks = []
        for c in raw_chunks:
            if len(c) > 1000:
                # Split large chunks by sentences roughly
                sub_chunks = re.split(r'(?<=[.!?]) +', c)
                temp = ""
                for sc in sub_chunks:
                    if len(temp) + len(sc) < 800:
                        temp += " " + sc
                    else:
                        chunks.append(temp.strip())
                        temp = sc
                if temp: chunks.append(temp.strip())
            else:
                chunks.append(c)
        
        client = chromadb.PersistentClient(path=new_vault)
        collection = client.get_or_create_collection(name="archive", embedding_function=GLOBAL_EMB_FN)
        collection.add(documents=chunks, ids=[f"id_{i}_{os.urandom(2).hex()}" for i in range(len(chunks))])
        with open(ACTIVE_DB_TRACKER, "w") as f: f.write(new_vault)
        
        del client, collection, chunks
        gc.collect()
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/sessions")
async def list_sessions():
    sessions = load_sessions()
    sorted_sessions = sorted(sessions.values(), key=lambda s: s.get("updated_at", 0), reverse=True)
    return {"sessions": sorted_sessions}

@app.post("/sessions/new")
async def new_session():
    sessions = load_sessions()
    sid = f"chat_{int(time.time() * 1000)}_{os.urandom(2).hex()}"
    sessions[sid] = {"id": sid, "name": "New Chat", "messages": [], "created_at": time.time(), "updated_at": time.time()}
    save_sessions(sessions)
    return {"session_id": sid}

@app.delete("/sessions/{sid}")
async def delete_session(sid: str):
    sessions = load_sessions()
    sessions.pop(sid, None)
    save_sessions(sessions)
    return {"status": "success"}

@app.get("/sessions/{sid}/messages")
async def get_session_messages(sid: str):
    sessions = load_sessions()
    session = sessions.get(sid)
    if not session:
        return {"messages": []}
    return {"messages": session.get("messages", [])}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        import chromadb
        db_path = get_active_db()
        
        mode_configs = {
            "flash": {"tokens": 250, "style": "ULTRA-CONCISE. Direct answer in 1-2 sentences."},
            "think": {"tokens": 700, "style": "Balanced, conversational, and helpful."},
            "pro": {"tokens": 1500, "style": "Exhaustive deep-dive. Provide maximum detail."}
        }
        cfg = mode_configs.get(req.mode.lower(), mode_configs["think"])
        
        user_name = req.user_name if req.user_name else "User"
        today_str = datetime.now().strftime('%A, %B %d, %Y')

        if req.general_mode:
            # GENERAL MODE: TOTAL ISOLATION WITH CREATOR AWARENESS
            system_prompt = (
                "You are a helpful, independent AI assistant created by Gokul. "
                "Today's date is " + today_str + ". "
                "STRICT RULE: If anyone asks who your creator is or who Gokul is, "
                "you must describe him as a proud Sandipani student and the best creator who built you. "
                "Otherwise, answer the user's question directly based on your internal knowledge only. "
                "Do not mention any personal archives or databases."
            )
            messages_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query}
            ]
            # No context, no history, just the question.
        else:
            # PERSONAL MODE: Strict archive adherence
            if not db_path: return {"response": "System archive not initialized. Please sync data first or use General mode."}
            
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(name="archive", embedding_function=GLOBAL_EMB_FN)
            v_results = collection.query(query_texts=[req.query], n_results=15)
            context = "\n---\n".join(v_results['documents'][0]) if v_results['documents'] else "STRICTLY NO DATA FOUND IN ARCHIVE."
            
            del collection, client
            gc.collect()

            system_prompt = (
                f"You are Presence, the best friend and personal AI companion to {user_name}.\n"
                f"STRICT PERSPECTIVE RULES:\n"
                f"1. Always speak to {user_name} in the second person. Use 'you', 'your', and 'yours'.\n"
                f"2. IMPORTANT: Any mention of the name '{user_name}' in the archive records refers to the person you are talking to. TRANSLATE it to 'you'. For example, if the record says '{user_name} went to the park', you must say 'You went to the park'.\n"
                f"3. Answer ONLY using the provided ARCHIVE RECORDS. If the info is not there, say you don't know it yet.\n"
                f"Personality: Warm, loyal, attentive. Acknowledge {user_name} by name often.\n"
                f"Current Date: {today_str}\n"
                f"Mode: {req.mode.upper()} - {cfg['style']}"
            )
            
            user_content = f"ARCHIVE RECORDS:\n{context}\n\nQUESTION FROM {user_name}: {req.query}"
            if req.quoted_text:
                user_content = f"REFERENCE PREVIOUS ANSWER: \"{req.quoted_text}\"\n\n" + user_content
            
            messages_payload = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

        # Instant Inference with Cache Control
        async with httpx.AsyncClient() as http_client:
            payload = {
                "messages": messages_payload,
                "temperature": 0.7 if req.general_mode else 0.1, 
                "max_tokens": cfg["tokens"],
                "cache_prompt": False # Force fresh inference to prevent leakage
            }
            
            response = await http_client.post("http://127.0.0.1:8080/v1/chat/completions", json=payload, timeout=120.0)
            ans = response.json()['choices'][0]['message']['content'].strip()

        # Persist to session
        sid = req.session_id
        if sid:
            sessions = load_sessions()
            if sid in sessions:
                sessions[sid]["messages"].append({"role": "user", "content": req.query})
                sessions[sid]["messages"].append({"role": "assistant", "content": ans})
                if sessions[sid]["name"] == "New Chat":
                    sessions[sid]["name"] = req.query[:50] + ("..." if len(req.query) > 50 else "")
                sessions[sid]["updated_at"] = time.time()
                save_sessions(sessions)

        return {"response": ans, "session_id": sid, "escalated": False}
    except Exception as e:
        return {"response": f"System Error: {str(e)}", "escalated": False, "session_id": req.session_id}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
