from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Define LLM
llm_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b", 
    api_key=os.environ["GROQ_API_KEY"]
)

# Prompt template
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know. Do not make up an answer.
Don't use information outside the provided context.
Question: {question}
Context: {context}
Answer:
"""

# Reusable logic
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})
