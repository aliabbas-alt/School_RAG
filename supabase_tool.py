# supabase_tool.py
from langchain_core.tools import Tool
from storage.query_supabase import search_documents

def supabase_search_tool(query: str) -> str:
    results = search_documents(query=query, top_k=5, threshold=0.7)
    if not results:
        return "No relevant documents found."

    # Join summary snippets; adjust slicing as needed
    snippets = []
    for r in results:
        snippet = r.get("content", "")
        metadata = f"[source:{r.get('source')} page:{r.get('page')} score:{r.get('similarity'):.3f}]"
        snippets.append(f"{metadata} {snippet[:300]}...")
    return "\n\n".join(snippets)

supabase_tool = Tool.from_function(
    name="supabase_search", 
    description="Search educational documents stored in Supabase using semantic similarity. Returns short snippets with sources.",
    func=supabase_search_tool
)