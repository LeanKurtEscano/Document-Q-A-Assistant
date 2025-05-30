from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunking:
    
    @staticmethod
    def split_into_parts(text: str, part_size: int = 5000) -> list[str]:
    
        return [text[i:i + part_size] for i in range(0, len(text), part_size)]

    @staticmethod
    def recursive_character_base_chunking(text: str, chunk_size: int = 600, chunk_overlap: int = 150):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [{"text": chunk, "index": i} for i, chunk in enumerate(chunks)]

    @staticmethod
    def parallel_chunking(text: str, part_size=5000, max_workers=4):
        parts = Chunking.split_into_parts(text, part_size=part_size)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(Chunking.recursive_character_base_chunking, parts))
       
        all_chunks = []
        idx = 0
        for chunk_list in results:
            for chunk in chunk_list:
                chunk["index"] = idx
                all_chunks.append(chunk)
                idx += 1
                
        return all_chunks
