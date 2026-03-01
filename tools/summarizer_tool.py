from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@tool
def summarizer_tool(text: str) -> str:
    """Summarize a long piece of text into concise key points."""
    if len(text) < 200:
        return text

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    prompt = (
        "Summarize the following text into clear, concise bullet points. "
        "Preserve all important facts, numbers, and conclusions.\n\n"
        f"Text:\n{text[:8000]}"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"Summarization failed: {e}"
