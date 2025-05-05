from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
import requests
import urllib.parse

# --- STEP 1: Setup REST API Tool for QuestDB ---

def query_questdb_restapi(sql_query: str) -> str:
    print("sql quer: ", sql_query)
    questdb_ip = "40.81.240.69"  # Your QuestDB IP
    questdb_port = "9000"         # Your QuestDB Port
    url = f"http://{questdb_ip}:{questdb_port}/exec"
    
    params = {"query": sql_query}
    encoded_params = urllib.parse.urlencode(params)
    full_url = f"{url}?{encoded_params}"

    response = requests.get(full_url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Error {response.status_code}: {response.text}"

# --- STEP 2: Connect to local LLaMA 3 ---
llama_3_local = ChatOllama(
    base_url="http://13.201.0.207:11434",  # Your Ollama Server IP:Port
    model="llama3.1",
    temperature=0.1
)

# --- STEP 3: Define Table Descriptions ---
table_descriptions = {
    "electronics": "phones, laptops, TVs with price and quantity.",
    "food": "packaged food items, snacks, processed foods with prices.",
    "vegetables": "fresh vegetables and fruits with prices and stock."
}

# --- STEP 4: Create Tools for Each Table ---
def make_sql_tool(table_name: str, description: str):
    return Tool(
        name=f"Query_{table_name.capitalize()}_Table",
        description=f"Use this tool when the user asks about {description}. Input should be a SQL WHERE clause condition.",
        func=lambda query: query_questdb_restapi(f"SELECT * FROM {table_name} WHERE {query}")
    )

electronics_tool = make_sql_tool("electronics", table_descriptions["electronics"])
food_tool = make_sql_tool("food", table_descriptions["food"])
vegetables_tool = make_sql_tool("vegetables", table_descriptions["vegetables"])

tools = [electronics_tool, food_tool, vegetables_tool]

# --- STEP 5: Create Prompt with better instructions ---
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful agent that chooses one or more table tools based on the user's needs. "
     "Sometimes the user may need products from multiple tables. "
     "Select the appropriate table(s) and combine results if necessary."
    ),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

# --- STEP 6: Create Router Agent ---
router_agent = create_openai_functions_agent(
    llm=llama_3_local,
    tools=tools,
    prompt=prompt
)

router_executor = AgentExecutor(
    agent=router_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# --- STEP 7: Main Query Function ---
def supermarket_query(user_input: str):
    tool_response = router_executor.invoke({"input": user_input})
    print(f"\n[Router Picked Tool Response]:\n{tool_response}\n")
    return tool_response

# --- STEP 8: Example Random Queries ---
if __name__ == "__main__":
    queries = [
        "Which are the fruits availible ",
        "Get me fruits under 5 dollars",
        "Give me phones and snacks available under 100 dollars",
        "I want a cart of fruits, food, and electronics worth 200 dollars",
        "Get me fresh vegetables and laptops under 50 dollars",
        "Give me any 5 cheapest products overall",
        "Get me 2 snacks, 1 phone, and 3 vegetables under 300 dollars",
        "List electronics and fruits that cost below 150 dollars"
    ]

    for idx, q in enumerate(queries, 1):
        print(f"\n----- Query {idx}: {q} -----")
        result = supermarket_query(q)
        print(result)
        break
    
