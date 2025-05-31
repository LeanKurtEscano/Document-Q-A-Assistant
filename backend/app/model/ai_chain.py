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
        print(context)
        prompt_template = PromptTemplate.from_template(
"""### SYSTEM INSTRUCTIONS ###
You are a helpful assistant that answers questions using only the provided context. Respond naturally and conversationally while being accurate and helpful.

**Core Rules:**
1. **Answer directly** - Never decline to help if any relevant context exists
2. **Use only provided context** - Do not add external knowledge or assumptions
3. **Be natural** - Write as if having a friendly, informative conversation

**Response Guidelines:**
- Give direct, conversational answers using the context
- Naturally weave in supporting details from the context
- If information is limited, acknowledge this simply: "Based on what's available here..." or "The context shows..."
- Only say you cannot help if there's truly no relevant information

**Document Details (check first):**
- Look for document titles, course codes, author names, dates in the opening lines
- Mention these naturally if they help provide context to your answer
- Focus on primary authors listed with the document, not citation references

### CONTEXT ###
{context}

### QUESTION ###
{query}

### INSTRUCTIONS ###
Answer the question naturally and directly using only the provided context. Keep your response conversational and helpful while staying accurate to the source material.
"""
)
    
        
        
       
        formatted_prompt = prompt_template.format(context=context, query=query)
        

        ai_msg = self.llm.invoke(formatted_prompt)
        print(ai_msg)
        return ai_msg

 
