# Arogya_AI Backend Engine

Core advanced RAG triage engine with voice transcription and multi-turn conversational memory.

## 🚀 Local Setup Instructions
1. Navigate to the backend: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file inside the `backend/` folder and add your key:
   `GOOGLE_API_KEY=your_gemini_api_key`
4. Run the API server: `python app.py`

## 📡 Active API Endpoints
* **POST** `/api/triage` - Expects JSON body `{"transcript": "text", "session_id": "id"}`
* **POST** `/api/triage-audio` - Expects multipart form-data with an audio file stream under the key `file`.