# query_supabase.py
"""
Query stored embeddings using similarity search.
"""

import os
from dotenv import load_dotenv
from storage.supabase_storage import SupabaseVectorStore
from embeddings.registry import build_provider

load_dotenv()


def search_documents(
    query: str,
    school_id: int = None,
    curriculum_type: str = None,
    document_type: str = None,
    top_k: int = 5,
    threshold: float = 0.75
):
    """
    Search documents using natural language query.
    
    Args:
        query: Natural language search query
        school_id: Filter by school ID
        curriculum_type: Filter by curriculum
        document_type: Filter by document type
        top_k: Number of results to return
        threshold: Minimum similarity threshold
    """
    print(f"\n{'='*60}")
    print(f"SIMILARITY SEARCH")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"School ID: {school_id or 'All'}")
    print(f"Curriculum: {curriculum_type or 'All'}")
    print(f"Document Type: {document_type or 'All'}")
    print(f"{'='*60}\n")
    
    # Generate query embedding
    print("🔮 Generating query embedding...")
    provider = build_provider("openai", model="text-embedding-3-small")
    query_result = provider.embed_texts([query])
    query_embedding = query_result.vectors[0]
    
    # Search Supabase
    vector_store = SupabaseVectorStore()
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        match_threshold=threshold,
        match_count=top_k,
        school_id=school_id,
        curriculum_type=curriculum_type,
        document_type=document_type
    )
    
    # Display results
    if not results:
        print("❌ No results found")
        return
    
    print(f"\n📄 Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{'─'*60}")
        print(f"Result {i} - Similarity: {result['similarity']:.3f}")
        print(f"{'─'*60}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"School ID: {result['school_id']}")
        print(f"Curriculum: {result['curriculum_type']}")
        print(f"Document Type: {result['document_type']}")
        print(f"\nContent Preview:")
        print(f"{result['content'][:300]}...")
        print()


def main():
    """Interactive query interface."""
    print("="*60)
    print("DOCUMENT SEARCH INTERFACE")
    print("="*60)
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Optional filters
        school_id_input = input("Filter by school ID (press Enter to skip): ").strip()
        school_id = int(school_id_input) if school_id_input else None
        
        curriculum_input = input("Filter by curriculum (press Enter to skip): ").strip()
        curriculum_type = curriculum_input.upper() if curriculum_input else None
        
        try:
            search_documents(
                query=query,
                school_id=school_id,
                curriculum_type=curriculum_type,
                top_k=3,
                threshold=0.70
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
