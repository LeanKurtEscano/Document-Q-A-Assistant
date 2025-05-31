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
You MUST answer using ONLY the provided context. Follow these rules:
1. **NEVER DECLINE** if context exists - even partial answers are acceptable
2. If context answers the question: 
   - Provide a concise response 
   - Include VERBATIM quotes supporting your answer
3. If context partially answers:
   - State "Based on the context: [partial answer]"
   - Explain what's missing
4. ONLY say "I cannot answer" if the context contains ZERO relevant information

### CONTEXT ###
{context}

### USER QUESTION ###
{query}

### RESPONSE FORMAT ###

Write your response as follows:

- Start with a direct answer to the question, integrating evidence naturally where possible.

- If you used a specific quote, add it at the end in parentheses with the note "Source: [quote]".

- If the context is partial, mention that the information might be incomplete.

- Do not use markdown, headers, or numbered lists in the response"""
    )
        
        
        # Format the prompt with both context and query
        formatted_prompt = prompt_template.format(context=context, query=query)
        
        # Invoke the LLM with the formatted prompt
        ai_msg = self.llm.invoke(formatted_prompt)
        print(ai_msg)
        return ai_msg

 
