import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
DB_DIR = "./chroma_db"

# --- GLOBAL SESSION MEMORY STORAGE ---
# Tracks the ongoing conversation for each unique phone session
SESSION_STORAGE = {}

def identify_clinical_router(user_transcript):
    text = user_transcript.lower()
    if any(k in text for k in ["child", "baby", "pediatric", "convulsing", "seizure", "dehydration", "etat", "infant", "hot", "fever", "sick"]):
        return {"clinical_domain": "pediatric_emergency_triage"}
    elif any(k in text for k in ["mother", "delivery", "bleeding", "pregnancy", "childbirth", "asha", "maternal", "labour"]):
        return {"clinical_domain": "safe_childbirth_practices"}
    elif any(k in text for k in ["pulmonary", "cough", "lung", "breathing", "tb", "covid", "asthma", "cold", "respiratory"]):
        return {"clinical_domain": "pulmonary_respiratory_safety"}
    elif any(k in text for k in ["elderly", "geriatric", "falls", "senior", "aged", "old age", "grandfather", "grandmother"]):
        return {"clinical_domain": "geriatric_patient_safety"}
    return None

def run_arogya_triage(user_query, session_id="default_session"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # 1. Fetch or initialize chat history string for this session
    if session_id not in SESSION_STORAGE:
        SESSION_STORAGE[session_id] = []
        
    chat_history_list = SESSION_STORAGE[session_id]
    chat_history_str = "\n".join([f"{msg['role'].upper()}: {msg['text']}" for msg in chat_history_list])
    
    # Combine history + new query for router intelligence
    combined_context_query = f"{chat_history_str}\nUSER: {user_query}"
    metadata_filter = identify_clinical_router(combined_context_query)
    print(f"\n⚡ [Router Gate] Active Metadata Filter: {metadata_filter} | Session: {session_id}")
    
    retriever_kwargs = {"k": 3}
    if metadata_filter:
        retriever_kwargs["filter"] = metadata_filter
        
    retriever = vector_store.as_retriever(search_kwargs=retriever_kwargs)
    
    # UPGRADED SYSTEM PROMPT HANDLING CONVERSATIONAL MEMORY & CLARIFICATION
    system_prompt = (
        "You are ArogyaAI, an advanced rural healthcare triage system.\n"
        "You must respond ONLY with a single valid JSON object. Do not include markdown wraps.\n\n"
        "JSON keys required:\n"
        "1. 'priority_level': 'CRITICAL', 'URGENT', 'NON-URGENT', or 'UNKNOWN'.\n"
        "2. 'requires_navigation': true or false.\n"
        "3. 'speech_bubble_text': Instruction or clarification question for the user.\n"
        "4. 'first_aid_steps': List of clinical actions from context (empty if clarifying).\n"
        "5. 'status_code': 'SUCCESS', 'CLARIFICATION_NEEDED', 'OUT_OF_SCOPE', or 'UNVERIFIED'.\n\n"
        "INTERACTION PROTOCOLS:\n"
        "- CLARIFICATION RULE: If the user query or history is vague (e.g., 'my child is sick' or 'I feel bad') "
        "and lacks specific metrics or symptoms needed to match a protocol, set status_code to 'CLARIFICATION_NEEDED', "
        "requires_navigation to false, priority_level to 'UNKNOWN', and use speech_bubble_text to ask a clear, "
        "compassionate follow-up question asking for specific symptoms (like fever, bleeding, or breathing shifts).\n"
        "- If exact danger symptoms are clear in the history, execute standard triage logic immediately.\n\n"
        "Ongoing Conversation History:\n{chat_history}\n\n"
        "Verified Medical Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0.0,
        google_api_key="GOOGLE_API_KEY",
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Inject both current input and string history into the template execution
    response = rag_chain.invoke({
        "input": user_query,
        "chat_history": chat_history_str if chat_history_str else "No prior exchange."
    })
    
    try:
        structured_json = json.loads(response["answer"])
        
        # Save this successful loop state into memory
        SESSION_STORAGE[session_id].append({"role": "user", "text": user_query})
        SESSION_STORAGE[session_id].append({"role": "model", "text": structured_json.get("speech_bubble_text", "")})
        
        return structured_json
    except Exception as parse_error:
        print(f"⚠️ JSON Fallback: {str(parse_error)}")
        return {
            "priority_level": "UNKNOWN",
            "requires_navigation": False,
            "speech_bubble_text": "Could you please describe the specific symptoms you are experiencing?",
            "first_aid_steps": [],
            "status_code": "CLARIFICATION_NEEDED"
        }

if __name__ == "__main__":
    print("🏥 ArogyaAI Triage Engine Initializing Test Protocols...")
    
    sample_voice_input_1 = "A two year old child is brought in acutely convulsing and feels very hot"
    print(f"\n🗣️ User Speech Input: '{sample_voice_input_1}'")
    answer_1 = run_arogya_triage(sample_voice_input_1)
    print(f"🤖 ArogyaAI Audio Output:\n{answer_1}")