import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from supabase_tool import supabase_tool  # your custom tool

# Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def build_agent():
    # Define your LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    # Define your tools
    tools = [supabase_tool]

    # Optional system message
    system_prompt = """You are a helpful educational assistant.
Use the given tools to answer questions accurately about documents."""

    # ✅ Correct call — no 'verbose'
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent

def run_loop():
    agent = build_agent()
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        query = input("Ask: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        try:
            # ✅ New agents use invoke()
            result = agent.invoke({"input": query})
            print("\n--- Agent response ---\n")
            print(result.get("output", result))
            print("\n----------------------\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    run_loop()
