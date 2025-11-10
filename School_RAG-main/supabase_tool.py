# supabase_tool.py
"""
LangChain tool for semantic search - Optimized for textbook content.
"""

from langchain_core.tools import Tool
from storage.supabase_storage import SupabaseVectorStore
from embeddings.registry import build_provider
import os


def search_school_documents(query: str) -> str:
    """
    Search educational documents using semantic similarity.
    Optimized for finding textbook content and activities.
    """
    try:
        # Generate query embedding
        provider = build_provider("openai", model="text-embedding-3-small")
        query_result = provider.embed_texts([query])
        query_embedding = query_result.vectors[0]
        
        # Search with RELAXED threshold for better textbook matching
        vector_store = SupabaseVectorStore()
        results = vector_store.similarity_search(
            query_embedding=query_embedding,
            match_threshold=0.40,  # ✅ LOWERED to 0.40 for better textbook matching
            match_count=10,  # ✅ Get more results
            school_id=None,
            curriculum_type=None,
            document_type=None,
            academic_year=None
        )
        
        if not results:
            return """No relevant documents found. This could mean:
1. The content hasn't been uploaded yet
2. Try rephrasing your question
3. The similarity threshold might need adjustment"""
        
        # Format results with more context
        formatted_results = []
        for i, result in enumerate(results, 1):
            similarity = result.get('similarity', 0)
            source = result.get('source', 'Unknown')
            # Extract just the filename from full path
            if '\\' in source:
                source = source.split('\\')[-1]
            elif '/' in source:
                source = source.split('/')[-1]
            
            page = result.get('page', 'N/A')
            content = result.get('content', '')
            doc_type = result.get('document_type', 'N/A')
            curriculum = result.get('curriculum_type', 'N/A')
            
            # Show more content for textbooks
            content_preview = content[:600] if len(content) > 600 else content
            
            formatted_result = f"""
---
**Result {i}** (Match Score: {similarity:.1%})
- **Source**: {source}
- **Page**: {page}
- **Type**: {doc_type} | **Curriculum**: {curriculum}
- **Content**:
{content_preview}...
"""
            formatted_results.append(formatted_result.strip())
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Search error: {str(e)}\nPlease check your Supabase connection."


# Create the LangChain Tool
supabase_tool = Tool.from_function(
    name="search_school_documents",
    description="""Search ALL educational documents including textbooks, handbooks, and curriculum materials.

This tool searches for:
- Textbook content (stories, chapters, lessons, activities)
- Exercises and questions from textbooks
- Handbooks and policies
- Curriculum materials

Use for questions about:
- Story content ("What happens in [story name]?")
- Activities and exercises ("What are the activities in [chapter]?")
- Character information ("Who is Jahnavi?")
- Learning concepts and explanations
- Any content from uploaded documents

Works best with specific names, titles, or topics from the documents.""",
    func=search_school_documents,
    return_direct=False
)
