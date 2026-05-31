import os
import logging
from src.llm.prompt_templates import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

def generate_answer(context: str, query: str) -> str:
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
        )

        user_message = (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER (based only on the context above):"
        )

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=0.0,
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"Groq responded: {answer[:80]}...")
        return answer

    except Exception as e:
        logger.error(f"Groq request failed: {e}")
        return _fallback_answer(context, query)


def _fallback_answer(context: str, query: str) -> str:
    if not context or not context.strip():
        return "Not found in documents."
    query_words = set(query.lower().split())
    matches = sum(1 for word in query_words if word in context.lower())
    if matches > 0:
        return f"Based on the documents: {context[:400]}..."
    return "Not found in documents."