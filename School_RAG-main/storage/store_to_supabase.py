# storage/store_to_supabase.py
"""
Main script for storing document embeddings in Supabase.
Run after processing PDFs with run_parse.py.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.runner import embed_and_store_supabase
from storage.supabase_storage import SupabaseVectorStore

# Load environment variables
load_dotenv()


def main():
    """Interactive script for storing embeddings in Supabase."""
    print("="*70)
    print("  SUPABASE EMBEDDING STORAGE FOR SCHOOL DOCUMENTS")
    print("="*70)
    
    # Step 1: Get JSON file path
    print("\n📄 Step 1: Locate Your Processed JSON File")
    print("-" * 70)
    json_path = input("Enter path to JSON file (from run_parse.py): ").strip()
    
    if not os.path.exists(json_path):
        print(f"❌ ERROR: File not found: {json_path}")
        print("   Please run 'python run_parse.py' first to generate a JSON file")
        return
    
    print(f"✅ Found: {json_path}")
    
    # Step 2: School information
    print("\n🏫 Step 2: Enter School Information")
    print("-" * 70)
    
    try:
        school_id = int(input("School ID (number): ").strip())
    except ValueError:
        print("❌ ERROR: School ID must be a number")
        return
    
    curriculum_type = input("Curriculum Type (CBSE/SSE/ICSE/IB/etc): ").strip().upper()
    
    if not curriculum_type:
        print("❌ ERROR: Curriculum type is required")
        return
    
    # Step 3: Document metadata
    print("\n📚 Step 3: Enter Document Metadata")
    print("-" * 70)
    
    document_type = input("Document Type [curriculum]: ").strip() or "curriculum"
    academic_year = input("Academic Year [2025-26]: ").strip() or "2025-26"
    
    # Step 4: Confirmation
    print("\n" + "="*70)
    print("  CONFIGURATION SUMMARY")
    print("="*70)
    print(f"📄 JSON File:       {json_path}")
    print(f"🏫 School ID:       {school_id}")
    print(f"📖 Curriculum:      {curriculum_type}")
    print(f"📚 Document Type:   {document_type}")
    print(f"📅 Academic Year:   {academic_year}")
    print("="*70)
    
    confirm = input("\n⚠️  Proceed with storage? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("❌ Operation cancelled")
        return
    
    # Step 5: Execute storage
    print("\n🚀 Starting storage process...\n")
    
    try:
        result = embed_and_store_supabase(
            json_path=json_path,
            school_id=school_id,
            curriculum_type=curriculum_type,
            document_type=document_type,
            academic_year=academic_year,
            provider_name="openai",
            model="text-embedding-3-small",
            dimensions=None  # Use default 1536
        )
        
        # Success summary
        print("\n" + "="*70)
        print("  ✅ STORAGE SUCCESSFUL!")
        print("="*70)
        print(f"📊 Chunks Stored:     {result['inserted_count']}")
        print(f"🔮 Model Used:        {result['embedding_model']}")
        print(f"📏 Dimensions:        {result['embedding_dimension']}")
        print(f"🆔 First Record ID:   {result['inserted_ids'][0]}")
        print(f"🆔 Last Record ID:    {result['inserted_ids'][-1]}")
        print("="*70)
        
        # Show database statistics
        print("\n📈 Database Statistics (for your school):")
        print("-" * 70)
        vector_store = SupabaseVectorStore()
        stats = vector_store.get_document_stats(school_id=school_id)
        
        print(f"Total Documents:      {stats.get('total_documents', 0)}")
        print(f"Total Chunks:         {stats.get('total_chunks', 0)}")
        print(f"Avg Chunk Length:     {stats.get('avg_chunk_length', 0):.0f} characters")
        print("="*70)
        
        print("\n✅ Your embeddings are now searchable in Supabase!")
        print("   You can now use similarity search to query this data.")
        
    except Exception as e:
        print("\n" + "="*70)
        print("  ❌ ERROR OCCURRED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("1. ✓ Supabase credentials in .env file")
        print("2. ✓ Database tables created (run SQL setup)")
        print("3. ✓ OpenAI API key is valid")
        print("4. ✓ JSON file contains valid chunks")
        print("="*70)
        
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
