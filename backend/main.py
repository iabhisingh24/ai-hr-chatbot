import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import List, Optional
import requests
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
from fastapi import UploadFile, File
import shutil
import uuid

# Add this class for request model
class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, str]] = None
    top_k: Optional[int] = 5
    follow_up_context: Optional[str] = None

# Load environment variables
load_dotenv()

# ---------------- INITIALIZE FASTAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- LOAD AND PROCESS DOCUMENTS WITH METADATA ----------------
def load_documents_with_metadata():
    """Load all policy documents with proper metadata"""
    
    document_metadata = {
        "leave_policy.txt": {
            "doc_id": "leave_policy_v1",
            "title": "Leave Policy",
            "version": "1.0", 
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "Leave",
            "owner": "HR Department",
            "url": "/policies/leave_policy_v1.pdf"
        },
        "exit_policy.txt": {
            "doc_id": "exit_policy_v1",
            "title": "Exit Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN", 
            "category": "Exit",
            "owner": "HR Department",
            "url": "/policies/exit_policy_v1.pdf"
        },
        "benefits_policy.txt": {
            "doc_id": "employee_benefits_v1", 
            "title": "Employee Benefits Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "Benefits",
            "owner": "HR Department",
            "url": "/policies/employee_benefits_v1.pdf"
        },
        "hr_policy.txt": {
            "doc_id": "hr_policy_v1",
            "title": "HR General Policy",
            "version": "1.0",
            "effective_date": "2024-01-01", 
            "region": "IN",
            "category": "General",
            "owner": "HR Department",
            "url": "/policies/hr_policy_v1.pdf"
        }
    }
    
    all_docs = []
    
    for filename, metadata in document_metadata.items():
        try:
            file_path = f"docs/{filename}"
            loader = TextLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata["source"] = file_path
                
            all_docs.extend(docs)
            print(f"Loaded: {filename} with metadata")
            
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    
    return all_docs

# Load all documents with metadata
all_docs = load_documents_with_metadata()

# Process documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()


# ---------------- AI-POWERED RESPONSES WITH OPENROUTER ----------------
def get_ai_response(question, context):
    """
    Use OpenRouter AI with exact prompt guardrails from assignment
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # If no API key, use fallback
    if not api_key:
        print("No OpenRouter API key found, using fallback")
        return get_smart_fallback(question, context)
    
    try:
        prompt = f"""You are a precise HR policy assistant. Answer only from the provided policy context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer only from the provided policy context.
- Cite at least one source section with doc id and section header.
- If the answer is not clearly in the context, say "I don't have that in policy" and suggest contacting HR.
- Keep the answer under 200 words unless asked for details.
- Never invent numbers or dates. Prefer exact language and effective dates.

Return a JSON object with this structure:
{{
    "answer": "concise answer based on context",
    "confidence": "high/medium/low",
    "suggestion": "if answer is unclear, suggest contacting HR"
}}

ANSWER:"""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json", 
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "HR Policy Assistant"
        }
        
        data = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1  
        }
        
        print("Calling OpenRouter AI with assignment prompt...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"🔍 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()
            print(f"AI Raw Response: {ai_response}")
            
            # Try to parse JSON response
            try:
                
                if "{" in ai_response and "}" in ai_response:
                    json_start = ai_response.index("{")
                    json_end = ai_response.rindex("}") + 1
                    json_str = ai_response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    answer = parsed_response.get("answer", "No answer provided")
                    confidence = parsed_response.get("confidence", "medium")
                    suggestion = parsed_response.get("suggestion", "")
                    
                    if suggestion and "I don't have that in policy" in answer:
                        final_answer = f"{answer} {suggestion}"
                    else:
                        final_answer = answer
                        
                    return final_answer
                else:
                    return ai_response
                    
            except json.JSONDecodeError:
                print("Failed to parse JSON from AI response, using raw response")
                return ai_response
                
        else:
            print(f"OpenRouter API error: {response.status_code}")
            print(f"Error response: {response.text}")
            return get_smart_fallback(question, context)
            
    except Exception as e:
        print(f"AI API error: {e}")
        return get_smart_fallback(question, context)

def get_smart_fallback(question, context):
    """
    Fallback responses when AI is not available
    """
    question_lower = question.lower()
    context_lower = context.lower()
    
    if any(word in question_lower for word in ['working hour', 'office hour']):
        return "Standard working hours are 9 AM to 6 PM with work-from-home options available twice per week."
    
    elif any(word in question_lower for word in ['leave', 'sick', 'vacation']):
        if 'carry forward' in question_lower:
            return "Sick leaves cannot be carried forward to the next year and will lapse at year-end."
        return "Employees receive 12 paid casual leaves and 12 sick leaves per year."
    
    elif any(word in question_lower for word in ['benefit', 'insurance', 'perk']):
        return "Benefits include health insurance, annual bonuses, and work-from-home options."
    
    elif any(word in question_lower for word in ['exit', 'resign', 'notice']):
        return "The exit policy requires a 30-day notice period with final settlement within 45 days."
    
    else:
        return f"I found relevant policy information about your question. For specific details, please review the policy documents or contact HR."

@app.post("/ask")
async def ask_question(request: AskRequest):  
    start_time = time.time()  
    
    try:
        question = request.question
        filters = request.filters
        top_k = request.top_k or 5
        follow_up_context = request.follow_up_context
        
        print(f" Received question: {question}")
        print(f" Filters: {filters}, Top K: {top_k}, Follow-up: {follow_up_context}")

        if not question.strip():
            return {
                "answer": "Please ask a valid question.",
                "citations": [],
                "policy_matches": [],
                "confidence": "low",
                "disclaimer": "If your contract specifies otherwise, the contract prevails.",
                "metadata": {
                    "latency_ms": 0,
                    "retriever_k": 0,
                    "model": "none"
                }
            }

        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        print(f"Retrieved {len(retrieved_docs)} document chunks")

        citations = []
        policy_matches = set()

        for doc in retrieved_docs:
            citation = {
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "section": doc.metadata.get("title", "Relevant Section"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "page": 1
            }
            citations.append(citation)
            
            category = doc.metadata.get("category", "General")
            policy_matches.add(f"{category} Policy")

        answer = get_ai_response(question, context)
        
        def calculate_confidence(question, retrieved_docs, context):
            question_lower = question.lower()
            context_lower = context.lower()
            
            low_confidence_keywords = [
                'salary', 'pay', 'compensation', 'bonus amount', 'exact', 
                'specific', 'how much', 'when will', 'date', 'holiday list',
                'paternity', 'maternity', 'tax', 'promotion', 'balance'
            ]
            
            if any(keyword in question_lower for keyword in low_confidence_keywords):
                return "low"
            
            question_words = set(question_lower.split())
            context_words = set(context_lower.split())
            matching_words = question_words.intersection(context_words)
            
            if len(matching_words) >= 3 and len(retrieved_docs) >= 2:
                return "high"
            elif len(matching_words) >= 1 and len(retrieved_docs) >= 1:
                return "medium"
            else:
                return "low"

        confidence = calculate_confidence(question, retrieved_docs, context)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Return full structured response
        return {
            "answer": answer,
            "citations": citations,
            "policy_matches": list(policy_matches),
            "confidence": confidence,
            "disclaimer": "If your contract specifies otherwise, the contract prevails.",
            "metadata": {
                "latency_ms": latency_ms,
                "retriever_k": len(retrieved_docs),
                "model": "openrouter-llama-3.2-3b"
            }
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        print(f"Error occurred: {e}")
        return {
            "answer": "An error occurred while processing your request.",
            "citations": [],
            "policy_matches": [],
            "confidence": "low",
            "disclaimer": "If your contract specifies otherwise, the contract prevails.",
            "metadata": {
                "latency_ms": latency_ms,
                "retriever_k": 0,
                "model": "error"
            }
        }
        
# ============ ADD NEW ENDPOINTS RIGHT HERE ============

@app.post("/ingest")
async def ingest_documents(file: UploadFile = File(None)):
    """Upload or re-index policies. Accept .pdf/.docx/.txt/.md files"""
    try:
        # If no file provided, re-index existing documents
        if not file:
            print("Re-indexing existing policy documents...")
            return await reindex_existing_documents()

        allowed_extensions = ['.pdf', '.docx', '.txt', '.md']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            return {
                "status": "error", 
                "message": f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
            }
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"uploads/{str(uuid.uuid4())}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File uploaded: {file.filename}")
        
        # Load and process the new document
        try:
            if file_extension == '.pdf':
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
            elif file_extension == '.docx':
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            else:  # .txt, .md
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path)
            
            docs = loader.load()
            
            # Add basic metadata for new document
            for doc in docs:
                doc.metadata.update({
                    "doc_id": f"uploaded_{os.path.splitext(file.filename)[0]}",
                    "title": os.path.splitext(file.filename)[0].replace('_', ' ').title(),
                    "version": "1.0",
                    "effective_date": str(datetime.now().date()),
                    "region": "IN",
                    "category": "Uploaded",
                    "owner": "HR Department",
                    "url": f"/uploads/{file.filename}",
                    "source": file_path
                })
            
            # Add to existing vector store
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(docs)
            
            # Update vector store with new documents
            global vector_store, retriever
            vector_store.add_documents(split_docs)
            retriever = vector_store.as_retriever()
            
            return {
                "status": "success",
                "message": f"Successfully ingested {len(split_docs)} chunks from {file.filename}",
                "file_name": file.filename,
                "chunks_processed": len(split_docs),
                "document_id": f"uploaded_{os.path.splitext(file.filename)[0]}"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing file: {str(e)}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def reindex_existing_documents():
    """Re-index all existing policy documents"""
    try:
        # Load all documents with metadata (reuse your existing function)
        all_docs = load_documents_with_metadata()
        
        # Process documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(all_docs)
        
        # Recreate vector store
        global vector_store, retriever
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        retriever = vector_store.as_retriever()
        
        return {
            "status": "success", 
            "message": f"Successfully re-indexed {len(split_docs)} document chunks from {len(all_docs)} files",
            "documents_processed": len(split_docs),
            "files_processed": len(all_docs)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/documents")
async def list_documents():
    """List all available policy documents with full metadata"""
    documents = [
        {
            "doc_id": "leave_policy_v1",
            "title": "Leave Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "Leave",
            "owner": "HR Department",
            "url": "/policies/leave_policy_v1.pdf"
        },
        {
            "doc_id": "exit_policy_v1",
            "title": "Exit Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "Exit",
            "owner": "HR Department",
            "url": "/policies/exit_policy_v1.pdf"
        },
        {
            "doc_id": "employee_benefits_v1",
            "title": "Employee Benefits Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "Benefits",
            "owner": "HR Department",
            "url": "/policies/employee_benefits_v1.pdf"
        },
        {
            "doc_id": "hr_policy_v1",
            "title": "HR General Policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
            "region": "IN",
            "category": "General",
            "owner": "HR Department",
            "url": "/policies/hr_policy_v1.pdf"
        }
    ]
    
    return {
        "documents": documents,
        "total_documents": len(documents)
    }
@app.post("/feedback")
async def submit_feedback(request: Request):
    """Store user feedback about answers"""
    try:
        data = await request.json()
        feedback_data = {
            "question": data.get("question", ""),
            "rating": data.get("rating", ""),
            "comment": data.get("comment", ""),
            "timestamp": str(datetime.now())
        }
        print(f"User feedback received: {feedback_data}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "feedback_id": f"fb_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/healthz")
async def health_check():
    """Health check endpoint for readiness/liveness probes"""
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "service": "HR Policy Assistant API",
        "version": "1.0"
    }
