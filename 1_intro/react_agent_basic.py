from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent

# Load environment variables (like API keys)
load_dotenv()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Initialize the Tavily search tool
search_tool = TavilySearchResults(search_depth="basic")

# Define tools to be used by the agent
tools = [search_tool]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Run the prompt and print the result
try:
    response = agent.invoke("give me a tweet about cristiano's failed saudi league career in dank humour")
    print("\nGenerated Tweet:\n", response)
except Exception as e:
    print(f"Error: {e}")
