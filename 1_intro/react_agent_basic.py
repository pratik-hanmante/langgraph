from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent

# Load .env file for API keys and configs
load_dotenv()

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True  # Optional, for better formatting
)

# --- Tool Setup ---
search_tool = TavilySearchResults(search_depth="basic")
tools = [search_tool]

# --- Agent Setup ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True  # Safer prompt execution
)

# --- Prompt Execution ---
prompt = "give me a tweet about cristiano's failed saudi league career in dank humour"

try:
    result = agent.invoke(prompt)

    # Print output neatly
    if isinstance(result, str):
        print("\n📝 Tweet Output:\n", result.strip())
    elif isinstance(result, dict) and 'output' in result:
        print("\n📝 Tweet Output:\n", result['output'].strip())
    else:
        print("\n⚠️ Unexpected response format:\n", result)

except Exception as e:
    print(f"\n❌ Error during execution:\n{str(e)}")
