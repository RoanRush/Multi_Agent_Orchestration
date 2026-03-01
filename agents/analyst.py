from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.calculator_tool import calculator_tool
from tools.summarizer_tool import summarizer_tool

ANALYST_SYSTEM_PROMPT = """You are an analyst agent. Your job is to analyze research data
and extract key insights.

- Break down complex information into clear key points
- Use the calculator tool when numerical analysis is needed
- Identify trends, risks, and opportunities
- Structure your analysis clearly (e.g., pros/cons, key findings)
"""


def get_analyst_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = [calculator_tool, summarizer_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYST_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)
