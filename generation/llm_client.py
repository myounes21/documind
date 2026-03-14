from functools import cache
from config import settings
from .prompt_templates import prompt


@cache
def _get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_llm_model,
        temperature=0
    )


@cache
def _get_openai_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_llm_model,
        temperature=0
    )


_LLM_PROVIDERS = {
    "groq": _get_groq_llm,
    "openai": _get_openai_llm,
}


@cache
def get_chain():
    from langchain_core.output_parsers import StrOutputParser

    llm_fn = _LLM_PROVIDERS.get(settings.llm_provider)

    if llm_fn is None:
        raise ValueError(
            f"Unknown LLM provider: '{settings.llm_provider}'. "
            f"Must be one of: {list(_LLM_PROVIDERS.keys())}"
        )

    llm = llm_fn()

    return prompt | llm | StrOutputParser()