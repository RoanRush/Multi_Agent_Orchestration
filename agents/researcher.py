from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.search_tool import search_tool
from tools.summarizer_tool import summarizer_tool

RESEARCHER_SYSTEM_PROMPT = """You are a research agent. Your job is to gather accurate,
relevant information on any topic you are given.

- Use the search tool to find information
- Use the summarizer tool to condense large texts
- Focus on facts and credible sources
- Always note where information came from when possible
"""


def get_researcher_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    tools = [search_tool, summarizer_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCHER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)
