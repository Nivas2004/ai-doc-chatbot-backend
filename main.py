from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ollama import chat
import os
import PyPDF2
from docx import Document
os.makedirs("docs", exist_ok=True)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_NAME = "llama2"
MAX_CHARS = 2000  
session_docs = {}  
def extract_text(file_path: str, filename: str) -> str:
    text = ""
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif filename.endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text
    return text
def chunk_text(text: str, max_chars: int = MAX_CHARS):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars
    return chunks
@app.post("/upload/")
async def upload_document(session_id: str = Form(...), file: UploadFile = File(...)):
    """Upload document once per session"""
    file_location = f"docs/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    content = extract_text(file_location, file.filename)
    if not content.strip():
        return {"status": "error", "message": "Document empty or could not extract text."}
    session_docs[session_id] = chunk_text(content)
    return {"status": "success", "message": f"Document uploaded, {len(session_docs[session_id])} chunks created."}
@app.post("/chat/")
async def chat_with_doc(session_id: str = Form(...), query: str = Form(...)):
    """Ask question using the previously uploaded document"""
    if session_id not in session_docs:
        return {"answer": "No document found for this session. Please upload first."}
    chunks_to_send = session_docs[session_id][:3] 
    content_to_send = "\n".join(chunks_to_send)
    try:
        response = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"{content_to_send}\n\nQuestion: {query}"}]
        )
        if hasattr(response, "message") and response.message is not None:
            answer = response.message.content.strip()
        elif isinstance(response, str):
            answer = response.strip()
        else:
            answer = str(response)
    except Exception as e:
        answer = f"Error querying Ollama: {str(e)}"
    return {"answer": answer}
