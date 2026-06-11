import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Path configurations supporting Windows backslash structures cleanly
DATA_DIR = os.path.join("backend", "knowledge_base")
DB_DIR = "./chroma_db"

def parse_metadata_and_content(text):
    """
    Extracts the custom RAG METADATA key-values from the top of the file
    and strips it from the core text body to prevent embedding pollution.
    """
    metadata = {}
    content = text
    
    # Locate the RAG METADATA block boundary
    if "RAG METADATA" in text:
        # Find everything between metadata declaration and the main title lines
        meta_section = re.search(r'RAG METADATA(.*?)(?=\n\n|\n[A-Z][a-z])', text, re.DOTALL)
        if meta_section:
            meta_text = meta_section.group(1)
            # Find key-value pairs (e.g., clinical_domain: safe_childbirth_practices)
            matches = re.findall(r'(\w+)\s*:\s*([^\n]+)', meta_text)
            for key, val in matches:
                metadata[key.strip()] = val.strip()
            
            # Clean content by stripping the raw metadata block header
            content = text.replace(meta_section.group(0), "")
            
    return metadata, content

def build_knowledge_base():
    """
    Builds a local, persistent vector database supporting both PDF assets 
    and clean .txt training registries.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    processed_docs = []
    text_splitter = MarkdownTextSplitter(chunk_size=600, chunk_overlap=50)
    
    print("🚀 Team Rudra Ingestion Engine Starting...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: The directory '{DATA_DIR}' could not be located.")
        return

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        full_text = ""
        
        # Scenario A: Handle standard PDF assets
        if file_name.endswith(".pdf"):
            print(f"📄 Processing PDF: {file_name}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            
        # Scenario B: Handle your new clean RAG text assets smoothly
        elif file_name.endswith(".txt"):
            print(f"📝 Processing TXT Document: {file_name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            except Exception as e:
                print(f"⚠️ Could not read text file {file_name}: {str(e)}")
                continue
        else:
            continue # Skip any other background files
            
        if not full_text.strip():
            continue
            
        # Parse custom structural metadata headers if present
        custom_metadata, clean_content = parse_metadata_and_content(full_text)
        
        # Dynamic router alignment based on file titles
        if "childbirth" in file_name.lower() or "asha" in file_name.lower():
            custom_metadata.setdefault("clinical_domain", "safe_childbirth_practices")
        elif "pulmonary" in file_name.lower() or "respiratory" in file_name.lower():
            custom_metadata.setdefault("clinical_domain", "pulmonary_respiratory_safety")
        elif "elderly" in file_name.lower() or "geriatric" in file_name.lower():
            custom_metadata.setdefault("clinical_domain", "geriatric_patient_safety")
        elif "etat" in file_name.lower() or "triage" in file_name.lower() or "participant" in file_name.lower():
            custom_metadata.setdefault("clinical_domain", "pediatric_emergency_triage")
            
        custom_metadata.setdefault("safety_critical", "true")
        custom_metadata.setdefault("source_file", file_name)
        
        chunks = text_splitter.split_text(clean_content)
        for chunk in chunks:
            processed_docs.append(
                Document(page_content=chunk, metadata=custom_metadata.copy())
            )
                
    if not processed_docs:
        print("⚠️ Ingestion halted: No parsed document chunks found inside the folder.")
        return

    print(f"📦 Embedding {len(processed_docs)} chunks into local vector database...")
    vector_store = Chroma.from_documents(
        documents=processed_docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("✅ Local knowledge base successfully updated and locked!")

if __name__ == "__main__":
    build_knowledge_base()