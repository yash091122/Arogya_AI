import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr

# Force bypass of the OpenMP duplicate runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import your working memory-enabled triage function from ruery.py
from database.ruery import run_arogya_triage

app = FastAPI(
    title="ArogyaAI Core API Engine",
    description="Production backend running Team Rudra's Advanced Audio RAG Pipeline with Conversational Memory",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strict schema definitions for incoming frontend JSON packets
class TriageRequest(BaseModel):
    transcript: str
    session_id: str = "default_session"

@app.get("/")
def home():
    return {"status": "online", "system": "ArogyaAI Conversational Audio Engine Active"}

@app.post("/api/triage")
def execute_text_triage(request: TriageRequest):
    """
    Handles multi-turn text input tracking from the frontend interface.
    """
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")
    try:
        # Route the query string and its identifier to the RAG memory layer
        ai_structured_response = run_arogya_triage(request.transcript, session_id=request.session_id)
        return {
            "success": True,
            "input_transcript": request.transcript,
            "structured_triage": ai_structured_response
        }
    except Exception as e:
        print(f"❌ Text Triage Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/triage-audio")
def execute_audio_triage(
    session_id: str = Query("default_session", description="Unique session token for tracking conversation multi-turn context"),
    file: UploadFile = File(...)
):
    """
    Receives raw audio voice recordings from the frontend microphone layer,
    transcribes speech to text dynamically, and streams it to the routed context loops.
    """
    print(f"🎙️ Incoming Audio File Stream: {file.filename} | Session: {session_id}")
    
    # Create an explicit path to drop temporary incoming audio tracks safely
    temp_dir = "./temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # 1. Stream incoming multipart form bytes onto local machine storage
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Spin up the audio feature extraction engine
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_file_path) as source:
            print("🎛️ Isolating vocal spectrum frequencies and filtering noise floor...")
            audio_data = recognizer.record(source)
            
            print("🧠 Executing localized speech transcription arrays...")
            spoken_transcript = recognizer.recognize_google(audio_data)
            
        print(f"🗣️ Transcribed Text Result: '{spoken_transcript}'")
        
        # 3. Route the transcribed text along with its session ID to your memory RAG layers
        ai_structured_response = run_arogya_triage(spoken_transcript, session_id=session_id)
        
        # Free up the temporary file resource immediately
        os.remove(temp_file_path)
        
        # Return a unified, structural JSON payload perfectly mapped for Yash and Anushka!
        return {
            "success": True,
            "filename": file.filename,
            "transcribed_text": spoken_transcript,
            "structured_triage": ai_structured_response
        }
        
    except sr.UnknownValueError:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail="Vocal data unclear or audio input contains empty speech data profiles.")
    except Exception as e:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)
        print(f"❌ Server Audio Processing Layer Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio Engine Pipeline Disrupted: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)