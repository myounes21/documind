from langchain_groq import ChatGroq
from config import settings
from prompt_templates import prompt

llm = ChatGroq(
    api_key=settings.groq_api_key,
    model=settings.groq_llm_model,
    temperature=0
)

chain = prompt | llm