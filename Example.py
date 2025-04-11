#basic agent working
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

agent.run("What is the square root of 289 plus the current price of AAPL?")

#multi prebuilt langchain working
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.duckduckgo_search.tool import DuckDuckGoSearchResults

from langchain.agents import (
    create_tool_calling_agent,
    create_chat_agent,
    AgentExecutor,
)
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# 📦 Load Azure credentials
load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)

#  React Agent (searching)
search_tool = DuckDuckGoSearchResults()
react_agent_executor = create_react_agent(llm, tools=[search_tool])

#  Chat Agent (general conversation)
chat_agent = create_chat_agent(llm)
chat_executor = AgentExecutor(agent=chat_agent, tools=[], verbose=True)

#  SQL Agent (query conversion)
db = SQLDatabase.from_uri("sqlite:///example.db")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_tool_calling_agent(llm, tools=sql_toolkit.get_tools())
sql_executor = AgentExecutor(agent=sql_agent, tools=sql_toolkit.get_tools(), verbose=True)

#  Router Logic
def route_agent(user_input: str):
    lower = user_input.lower()
    message = HumanMessage(content=user_input)

    if "sql" in lower or "convert" in lower or "query" in lower:
        print("➡️ SQL Agent")
        return sql_executor.invoke({"input": user_input})

    elif "search" in lower or "latest" in lower or "find" in lower:
        print("➡️ ReAct Search Agent")
        return react_agent_executor.invoke({"messages": [message]})

    else:
        print("➡️ Chat Agent")
        return chat_executor.invoke({"input": user_input})


#  Testing
if __name__ == "__main__":
    prompts = [
        "Hi, how are you today?",
        "Search the latest in AI research",
        "Convert to SQL: list all employees joined after 2022",
    ]

    for prompt in prompts:
        print(f"\n You: {prompt}")
        response = route_agent(prompt)
        print(f"\n Agent Response:\n{response}")

#other prebuilt agents and use cases:


#1. create_react_agent- A reasoning + acting agent that thinks step-by-step and uses tools like search or calculator.

#2. create_tool_calling_agent- An agent that calls tools using OpenAI-style structured function calling, great for APIs and SQL.

#3. create_openai_functions_agent`**  Specifically built to work with OpenAI function calling, allowing precise tool execution via schemas.

#4. create_structured_chat_agent- Designed for multi-step workflows with structured tool use and memory, ideal for advanced pipelines.

#5. `create_chat_agent-A simple conversational agent for natural dialogue without tool usage, like a friendly chatbot.

#Rag concept can be used here with Ragretriever provided by langchain.


#7. Agent Memory
#Agents can maintain memory across steps or sessions:

#ConversationSummaryMemory: TL;DR of past chat

#BufferMemory: Full conversation log

#EntityMemory: Remembers specific entities

#for this can maitain a  thread_id 
#use of builtin toolkits 
#output parsing is made availble as well


