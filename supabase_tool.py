# tools/supabase_tool.py - FIXED VERSION
from langchain_core.tools import Tool
from storage.supabase_storage import SupabaseVectorStore
from embeddings.registry import build_provider

def search_school_documents(query: str) -> str:
    try:
        provider = build_provider("openai", model="text-embedding-3-small")
        query_result = provider.embed_texts([query])
        query_embedding = query_result.vectors[0]
        
        vector_store = SupabaseVectorStore()
        results = vector_store.similarity_search(
            query_embedding=query_embedding,
            match_threshold=0.25,  # ✅ LOWERED to 0.25
            match_count=30,  # ✅ INCREASED to 30
            school_id=None,
            curriculum_type=None,
            document_type=None,
            academic_year=None
        )
        
        if not results:
            return "No results found."
        
        # ✅ PRIORITIZE image descriptions
        image_results = [r for r in results if r.get('content_type') == 'image_description']
        text_results = [r for r in results if r.get('content_type') != 'image_description']
        
        # Put images first, then text
        prioritized = image_results[:5] + text_results[:5]
        
        print(f"\n{'='*70}")
        print(f"DEBUG: Total={len(results)}, Images={len(image_results)}, Text={len(text_results)}")
        print(f"{'='*70}")
        
        formatted = []
        for i, r in enumerate(prioritized, 1):
            page = r.get('page', 'N/A')
            content_type = r.get('content_type', 'text')
            content = r.get('content', '')
            
            print(f"\nResult {i}: Page {page}, Type: {content_type}, Length: {len(content)}")
            
            # ✅ FULL content for ALL types
            formatted.append(f"**Page {page}** [{content_type.upper()}]\n{content}\n")
        
        return "\n---\n".join(formatted)
    
    except Exception as e:
        return f"Error: {e}"

supabase_tool = Tool.from_function(
    name="search_school_documents",
    description="Search documents - PRIORITIZES image descriptions",
    func=search_school_documents,
    return_direct=False
)
