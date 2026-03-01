from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.summarizer_tool import summarizer_tool

WRITER_SYSTEM_PROMPT = """You are a technical writer agent. Your job is to take research and analysis
and produce a clear, well-structured response for the end user.

- Write in a clear, professional tone
- Use markdown formatting (headers, bullets) where appropriate
- Make sure the response directly addresses the original task
- Keep it comprehensive but avoid filler text
- End with key takeaways when relevant
"""


def get_writer_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [summarizer_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)
