# embeddings.py
"""
Convenience script to generate embeddings and optionally store in Supabase.
Combines PDF parsing, embedding generation, and storage.
"""

import os
from dotenv import load_dotenv
from embeddings.runner import embed_and_save, embed_and_store_supabase
from config import (
    DEFAULT_EMBED_PROVIDER,
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_DIMENSIONS,
    DEFAULT_EMBED_OUTPUT_EXT
)

load_dotenv()


def main():
    """Main entry point for embedding generation."""
    print("="*60)
    print("DOCUMENT EMBEDDING GENERATOR")
    print("="*60)
    
    # Get JSON file path
    json_path = input("\nEnter path to JSON file (from run_parse.py): ").strip()
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return
    
    # Ask for storage destination
    print("\nWhere do you want to store embeddings?")
    print("1. Local file (Parquet/CSV/JSON)")
    print("2. Supabase (vector database)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Local file storage
        print("\nGenerating embeddings for local storage...")
        try:
            output_path = embed_and_save(
                json_path=json_path,
                provider_name=DEFAULT_EMBED_PROVIDER,
                model=DEFAULT_EMBED_MODEL,
                dimensions=DEFAULT_EMBED_DIMENSIONS,
                out_ext=DEFAULT_EMBED_OUTPUT_EXT
            )
            print(f"\n✅ Embeddings saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "2":
        # Supabase storage
        print("\nConfiguring Supabase storage...")
        school_id = int(input("Enter school ID: ").strip())
        curriculum_type = input("Enter curriculum type (CBSE/SSE/ICSE/IB): ").strip().upper()
        document_type = input("Enter document type [curriculum]: ").strip() or "curriculum"
        academic_year = input("Enter academic year [2025-26]: ").strip() or "2025-26"
        
        try:
            result = embed_and_store_supabase(
                json_path=json_path,
                school_id=school_id,
                curriculum_type=curriculum_type,
                document_type=document_type,
                academic_year=academic_year,
                provider_name=DEFAULT_EMBED_PROVIDER,
                model=DEFAULT_EMBED_MODEL,
                dimensions=DEFAULT_EMBED_DIMENSIONS
            )
            
            print(f"\n✅ Stored {result['inserted_count']} chunks in Supabase")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
