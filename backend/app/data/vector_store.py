import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

from app.data.chunking import Chunking



class PineconeStore:
    def __init__(self, api_key: str, index_name: str, dimension: int = 1536, cloud: str = "aws", region: str = "us-east-1"):
        
        self.pc = Pinecone(api_key=api_key)
        self.hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2" ,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
        
)
        
        self.chunking = Chunking()

        
        # Check if index exists, create if it doesn't
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(index_name)
      

    
    def upsert_texts(self, text: str, namespace: str = ""):
        chunks =  [c["text"] for c in self.chunking.parallel_chunking(text)]
        embeddings = self.hf.embed_documents(chunks)
        
        # Create vectors with proper format for new Pinecone client
        vectors = [
            {
                "id": str(i),
                "values": embeddings[i],
                "metadata": {"text": chunks[i]}
            }
            for i in range(len(chunks))
        ]
        
        # Upsert in batches if there are many vectors
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
    
    def query(self, query_text: str, top_k: int = 5, namespace: str = ""):
        query_embedding = self.embed_model.embed_query(query_text)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return results
    
    def delete_all(self, namespace: str = ""):
        """Delete all vectors in the specified namespace"""
        self.index.delete(delete_all=True, namespace=namespace)
    
    def get_index_stats(self):
        """Get statistics about the index"""
        return self.index.describe_index_stats()


