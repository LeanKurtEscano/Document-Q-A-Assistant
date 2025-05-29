from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from langchain_ollama import OllamaLLM

class QABot:
    def __init__(self, hf_token: str = None):
        self.llm = OllamaLLM(
            model="llama3.2",
        )
    
    def generate_response(self, query: str, context: str):
        prompt_template = PromptTemplate.from_template(
            """You are a helpful assistant answering questions based solely on the given context below.
If the answer is not contained within the context, please respond with 'The Question is beyond my scope'.

Context: {context}

Question: {query}

Answer:"""
        )
        
        # Format the prompt with both context and query
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        # Invoke the LLM with the formatted prompt
        ai_msg = self.llm.invoke(formatted_prompt)
        print(ai_msg)
        return ai_msg

 
if __name__ == "__main__":
    bot = QABot()
    
    context = "Paris is the capital city of France, known for the Eiffel Tower and rich history."
    question = "What is the capital of France?"
    
    bot.generate_response(question, context)