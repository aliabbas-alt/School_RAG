# run_agent.py
"""
Simple Q&A system using LangChain with direct tool calling.
No complex agent dependencies required.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from supabase_tool import search_school_documents

# Load environment variables
load_dotenv()


def create_qa_system():
    """Create simple Q&A system with LLM + tool."""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are an intelligent educational assistant for a school management system.

You have access to a school documents database containing:
- School policies, handbooks, and guidelines
- Curriculum materials (CBSE, SSE, ICSE, IB, etc.)
- Student and parent information
- Academic procedures and rules

When answering questions:
1. First, search the database using the search_school_documents function
2. Provide clear, accurate answers based on the retrieved information
3. Always cite your sources with [Source: filename, Page: number]
4. If information is not found, say so clearly
5. Be concise but thorough

Remember to cite document sources and page numbers for all facts."""

    return llm, system_prompt


def main():
    """Main interactive loop."""
    
    print("="*70)
    print("  🎓 SCHOOL DOCUMENT Q&A SYSTEM")
    print("="*70)
    print("Direct LangChain Integration with Supabase Vector Search")
    print("\nAsk questions about school documents, policies, curriculum, etc.")
    print("Type 'exit', 'quit', or 'q' to leave.\n")
    print("="*70)
    
    try:
        llm, system_prompt = create_qa_system()
        print("✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCheck your .env file has OPENAI_API_KEY")
        return
    
    conversation = []
    
    while True:
        try:
            # Get user input
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔍 Searching database...")
            
            # Step 1: Search database
            search_results = search_school_documents(query)
            
            # Step 2: Build context for LLM
            context = f"""User Question: {query}

Database Search Results:
{search_results}

Based on the search results above, provide a clear and accurate answer to the user's question. Always cite the source documents."""

            # Step 3: Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            print("🤔 Generating answer...\n")
            
            response = llm.invoke(messages)
            answer = response.content
            
            # Display answer
            print("="*70)
            print("🤖 Assistant:")
            print("="*70)
            print(answer)
            print("="*70)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
