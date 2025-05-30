from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import os
import tempfile
from app.data.vector_store import PineconeStore
import pdfplumber
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()
pinecone_store = PineconeStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME", "pdf-documents"),
    dimension=768
)





router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are accepted."})
    
    temp_file_path = None
    
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
      
        text = ""
        
        with pdfplumber.open(temp_file_path) as pdf:
           text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "No text could be extracted from the PDF."})
        
        
        
        stats = pinecone_store.get_index_stats()
        namespaces = stats.get("namespaces", {})
        if "pdf_documents" in namespaces and namespaces["pdf_documents"]["vector_count"] > 0:
            pinecone_store.delete_all("pdf_documents")

        pinecone_store.upsert_texts(text ,namespace="pdf_documents")
       
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded and processed successfully",
                "filename": file.filename,
                "text_length": len(text)
            }
        )
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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