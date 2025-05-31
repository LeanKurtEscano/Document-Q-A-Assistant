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
   - add in the response if the context contain zero to small relevant information "Please provide a more specific or detailed question to help me assist you better."
4. ONLY say "I cannot answer" if the context contains ZERO relevant information

**Metadata Extraction Priority**:  
   - Always scan the first 10 lines of the context for:  
     - Document titles, course codes (e.g., "PC 2211").  
     - Author/contributor names(usually after the name or title of the document).
     - Version/date markers (e.g., "v2.01182024").  
   - Explicitly state these in the response *before* content analysis.  
   - Focus on the authors listed immediately after the document title or in the authors’ section if there are any. Ignore cited authors in references or bibliography.
   
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

 
