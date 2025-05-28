from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import os
import tempfile
import PyPDF2
from app.model.ai_chain import AIBot

from app.data.vector_store import PineconeStore

router = APIRouter()

chat_bot = AIBot(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_store = PineconeStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME", "pdf-documents"),
    dimension=1536
)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are accepted."})
    
    temp_file_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Extract text from PDF
        text = ""
        with open(temp_file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "No text could be extracted from the PDF."})
        
        # Store in Pinecone
        pinecone_store.upsert_texts(text)
        print(pinecone_store.get_index_stats)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and processed successfully",
                "filename": file.filename,
                "text_length": len(text)
            }
        )
        
    except Exception as e:
        print(f"{e}")
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/query")
async def query_documents(query: str, top_k: int = 5):
    try:
        if not query.strip():
            return JSONResponse(status_code=400, content={"error": "Query cannot be empty"})
        
        results = pinecone_store.query(query, top_k=top_k)
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": float(match.score),
                "text": match.metadata.get("text", "")
            })
            
        print(pinecone_store.get_index_stats)
        
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "results": formatted_results
            }
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Query failed: {str(e)}"})