# run_parse.py (ENHANCED with optional vision processing)
import json
import os
from datetime import datetime
from ingest.pdf_loader import load_pdf
from ingest.chunk_utils import chunk_documents, summarize_chunks


def document_to_dict(doc):
    """Convert a LangChain Document to a JSON-serializable dict."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata or {}
    }


def default_output_name(pdf_path: str, use_unstructured: bool, chunk_size: int, chunk_overlap: int, use_vision: bool = False) -> str:
    """Generate a sensible default output filename based on inputs."""
    base = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
    loader = "unstructured" if use_unstructured else "pypdf"
    vision_tag = "_vision" if use_vision else ""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}__{loader}__cs{chunk_size}_co{chunk_overlap}{vision_tag}__{timestamp}.json"


def process_with_vision(pdf_path: str, document_type: str = "educational textbook"):
    """
    Process PDF with GPT-4 Vision to extract detailed image descriptions.
    """
    print("\n🖼️  VISION PROCESSING ENABLED")
    print("="*70)
    print("⚠️  This will use GPT-4 Vision to analyze all images in the PDF")
    print("   Each page costs ~$0.01-0.02 (for gpt-4o-mini)")
    print("   This may take several minutes for large PDFs")
    print("="*70)
    
    confirm = input("\nProceed with vision processing? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Vision processing skipped")
        return []
    
    try:
        from ingest.vision_processor import VisionProcessor
        
        processor = VisionProcessor()
        image_data = processor.process_pdf_with_vision(pdf_path, document_type=document_type)
        
        # Convert to chunk format
        vision_chunks = []
        for img_info in image_data:
            vision_chunks.append({
                "page_content": img_info["description"],
                "metadata": {
                    "source": img_info["pdf_source"],
                    "page": img_info["page_number"],
                    "content_type": "image_description",
                    "source_type": "gpt4_vision_analysis",
                    "image_path": img_info["image_path"]
                }
            })
        
        return vision_chunks
    
    except ImportError:
        print("\n❌ Vision processing not available")
        print("   Install required packages: pip install pdf2image pillow")
        print("   Also need Poppler: https://github.com/oschwartz10612/poppler-windows/releases")
        return []
    except Exception as e:
        print(f"\n❌ Vision processing error: {e}")
        return []


def main():
    print("PDF Parsing & Chunking Tool")
    print("Type 'exit' to quit.\n")
    
    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if pdf_path.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not os.path.exists(pdf_path):
            print(f"Error: File not found: {pdf_path}\n")
            continue

        # Ask about math mode FIRST
        use_math_input = input("Process as Math document? (y/n) [n]: ").strip().lower()
        use_math = (use_math_input == "y")

        if use_math:
            # Route into math_parser only — skip other options
            from math_parser import process_math_pdf, process_math_images, default_output_name

            use_unstructured = False
            chunk_size, chunk_overlap = 900, 180

            print(f"\n{'='*70}")
            print("STEP 1: MATH TEXT EXTRACTION")
            print(f"{'='*70}")

            math_chunks = process_math_pdf(pdf_path, use_unstructured, chunk_size, chunk_overlap)
            vision_chunks = process_math_images(pdf_path)

            all_chunks = math_chunks + vision_chunks

            print(f"\n{'='*70}")
            print("SUMMARY (MATH MODE)")
            print(f"{'='*70}")
            print(f"Text chunks:   {len(math_chunks)}")
            print(f"Image chunks:  {len(vision_chunks)}")
            print(f"Total chunks:  {len(all_chunks)}")
            print(f"{'='*70}")

            # Preview
            preview = input("\nPreview first 3 chunks? (y/n) [n]: ").strip().lower()
            if preview == "y":
                for i, c in enumerate(all_chunks[:3], 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"Type: {c['metadata'].get('content_type', 'text')}")
                    print(f"Page: {c['metadata'].get('page')}")
                    print(f"Content: {c['page_content'][:200]}...\n")

            # Save
            output_name = default_output_name(pdf_path)
            output_path = input(f"Output file [{output_name}]: ").strip() or output_name

            payload = {
                "source_pdf": os.path.basename(pdf_path),
                "loader": "math_parser",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vision_processing": True,
                "text_chunks": len(math_chunks),
                "image_chunks": len(vision_chunks),
                "total_chunks": len(all_chunks),
                "chunks": all_chunks,
                "mode": "math"
            }

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print(f"\n✅ Saved {len(all_chunks)} math chunks to {output_path}")
                print("\n📥 Next step: Upload to Supabase")
                print(f"   python storage/store_to_supabase.py")
                print(f"   Then enter: {output_path}\n")

            except Exception as e:
                print(f"Error saving file: {e}\n")

            continue  # Skip normal text+vision flow

        # --- Normal text + vision flow (unchanged) ---
        use_vision_input = input("Process images with GPT-4 Vision? (y/n) [n]: ").strip().lower()
        use_vision = (use_vision_input == "y")

        use_unstructured_input = input("Use UnstructuredPDFLoader? (y/n) [n]: ").strip().lower()
        use_unstructured = (use_unstructured_input == "y")

        try:
            chunk_size_input = input("Enter chunk size (default 900): ").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input else 900

            chunk_overlap_input = input("Enter chunk overlap (default 180): ").strip()
            chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else 180
        except ValueError:
            print("Invalid input for chunk size or overlap. Using defaults.\n")
            chunk_size = 900
            chunk_overlap = 180


        
        # Ask about vision processing
        use_vision_input = input("Process images with GPT-4 Vision? (y/n) [n]: ").strip().lower()
        use_vision = (use_vision_input == "y")
        
        # Text extraction options
        use_unstructured_input = input("Use UnstructuredPDFLoader? (y/n) [n]: ").strip().lower()
        use_unstructured = (use_unstructured_input == "y")
        
        try:
            chunk_size_input = input("Enter chunk size (default 900): ").strip()
            chunk_size = int(chunk_size_input) if chunk_size_input else 900
            
            chunk_overlap_input = input("Enter chunk overlap (default 180): ").strip()
            chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else 180
        except ValueError:
            print("Invalid input for chunk size or overlap. Using defaults.\n")
            chunk_size = 900
            chunk_overlap = 180
        
        # STEP 1: Extract text
        print(f"\n{'='*70}")
        print("STEP 1: TEXT EXTRACTION")
        print(f"{'='*70}")
        print(f"Loading PDF: {pdf_path} ...")
        
        try:
            docs = load_pdf(pdf_path, use_unstructured=use_unstructured)
            
            if not docs:
                print("Error: No documents loaded from PDF. The file might be empty or corrupted.\n")
                continue
            
            print(f"Loaded {len(docs)} page(s).")
        except Exception as e:
            print(f"Error loading PDF: {e}\n")
            continue
        
        print(f"Chunking with size={chunk_size}, overlap={chunk_overlap} ...")
        try:
            text_chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            if not text_chunks:
                print("Error: No chunks created. The document might be empty.\n")
                continue
            
            summary = summarize_chunks(text_chunks)
            if summary is None or len(summary) != 3:
                num_chunks = len(text_chunks)
                min_len = min([len(c.page_content) for c in text_chunks]) if text_chunks else 0
                max_len = max([len(c.page_content) for c in text_chunks]) if text_chunks else 0
            else:
                num_chunks, min_len, max_len = summary
            
            print(f"✅ Created {num_chunks} text chunks (min={min_len}, max={max_len} chars).")
        except Exception as e:
            print(f"Error during chunking: {e}\n")
            continue
        
        # STEP 2: Vision processing (if enabled)
        vision_chunks = []
        if use_vision:
            doc_type = input("\nDocument type (e.g., 'CBSE English textbook'): ").strip() or "educational textbook"
            vision_chunks = process_with_vision(pdf_path, doc_type)
        
        # STEP 3: Combine chunks
        all_chunks = []
        
        # Add text chunks
        for c in text_chunks:
            all_chunks.append({
                "page_content": c.page_content,
                "metadata": {
                    **c.metadata,
                    "content_type": "text"
                }
            })
        
        # Add vision chunks
        all_chunks.extend(vision_chunks)
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Text chunks:   {len(text_chunks)}")
        print(f"Image chunks:  {len(vision_chunks)}")
        print(f"Total chunks:  {len(all_chunks)}")
        print(f"{'='*70}")
        
        # Preview
        preview = input("\nPreview first 3 chunks? (y/n) [n]: ").strip().lower()
        if preview == "y":
            for i, c in enumerate(all_chunks[:3], 1):
                chunk_dict = c if isinstance(c, dict) else document_to_dict(c)
                print(f"\n--- Chunk {i} ---")
                print(f"Type: {chunk_dict['metadata'].get('content_type', 'text')}")
                print(f"Source: {chunk_dict['metadata'].get('source')}, Page: {chunk_dict['metadata'].get('page')}")
                print(f"Content: {chunk_dict['page_content'][:200]}...\n")
        
        # Save
        output_name = default_output_name(pdf_path, use_unstructured, chunk_size, chunk_overlap, use_vision)
        output_path = input(f"Output file [{output_name}]: ").strip() or output_name
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        payload = {
            "source_pdf": os.path.basename(pdf_path),
            "loader": "unstructured" if use_unstructured else "pypdf",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "vision_processing": use_vision,
            "text_chunks": len(text_chunks),
            "image_chunks": len(vision_chunks),
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Saved {len(all_chunks)} chunks to {output_path}")
            
            # Show next steps
            if use_vision:
                print("\n📥 Next step: Upload to Supabase")
                print(f"   python storage/store_to_supabase.py")
                print(f"   Then enter: {output_path}")
            
            print()
        
        except Exception as e:
            print(f"Error saving file: {e}\n")


if __name__ == "__main__":
    main()
