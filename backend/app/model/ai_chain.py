from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import os


class AIBot:
    def __init__(self, api_key: str):
        self.llm = OpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY", api_key),
            model="gpt-3.5-turbo",
            temperature=0.7,
        )

    def generate_response(self, query:str, context:str):
        prompt_template = PromptTemplate.from_template(
           """You are a helpful assistant answering questions based solely on the given context below. 
If the answer is not contained within the context, please respond with 'The Question is beyond my scope'.

Context:
{context}

Question:
{query}

Answer:"""
        )
        
        prompt_template.invoke({"context": context})
        prompt_template.invoke({"query": query})
        
        ai_msg = self.llm.invoke(prompt_template)
        print(ai_msg)
       
