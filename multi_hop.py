from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
import requests
import urllib.parse

# === 1. QuestDB REST API Helper ===
def query_questdb_restapi(sql_query: str) -> str:
    questdb_ip = "40.81.240.69"
    questdb_port = "9000"
    url = f"http://{questdb_ip}:{questdb_port}/exec"

    params = {"query": sql_query}
    encoded_params = urllib.parse.urlencode(params)
    full_url = f"{url}?{encoded_params}"

    response = requests.get(full_url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error {response.status_code}: {response.text}"

# === 2. Connect to local LLaMA 3 ===
llama3_local = ChatOllama(
    base_url="http://13.201.60.184:11434",
    model="llama3.1",
    temperature=0.1
)

# === 3. Create Tools for Each Table ===
table_descriptions = {
    "electronics": "phones, laptops, TVs with price and quantity.",
    "food": "packaged food items, snacks, processed foods with prices.",
    "vegetables": "fresh vegetables and fruits with prices and stock."
}

def make_sql_tool(table_name: str, description: str):
    return Tool(
        name=f"Query_{table_name.capitalize()}_Table",
        description=f"Use this tool to find {description}. Input should be a SQL WHERE clause condition.",
        func=lambda query: query_questdb_restapi(f"SELECT * FROM {table_name} WHERE {query}")
    )

electronics_tool = make_sql_tool("electronics", table_descriptions["electronics"])
food_tool = make_sql_tool("food", table_descriptions["food"])
vegetables_tool = make_sql_tool("vegetables", table_descriptions["vegetables"])

tools = [electronics_tool, food_tool, vegetables_tool]

# === 4. Memory Buffer for Multi-Hop ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === 5. Corrected ReAct Prompt ===
prompt = ChatPromptTemplate.from_template("""
You are an expert data agent using tools to query a QuestDB database.

ALWAYS use this structure:

Thought: Think about what you should do.
Action: choose one tool name from below.
Action Input: the input to the tool.

(You can use multiple Thought/Action/Action Input steps.)

ONLY after you have all the needed data, then say:
Final Answer: <your complete answer based on tools' results>

Available tools you can use:
{tools}

Available tool names are:
{tool_names}

Begin!

Question: {input}
{agent_scratchpad}
""")

# === 6. Create the ReAct Agent ===
react_agent = create_react_agent(
    llm=llama3_local,
    tools=tools,
    prompt=prompt
)

# === 7. Agent Executor ===
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# === 8. Supermarket Multi-Hop Query Function ===
def supermarket_multihop_query(user_input: str):
    response = react_executor.invoke({"input": user_input})
    print(f"\n[Agent Final Response]:\n{response}\n")
    return response

# === 9. Example Usage ===
if __name__ == "__main__":
    query1 = "Find fruits that cost less than the average price of electronics"
    query2 = "Get me vegetables cheaper than the cheapest phone available"

    print("\n----- Query 1 -----")
    result1 = supermarket_multihop_query(query1)

    print("\n----- Query 2 -----")
    result2 = supermarket_multihop_query(query2)






