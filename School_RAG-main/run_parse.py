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

def default_output_name(pdf_path: str, use_unstructured: bool, chunk_size: int, chunk_overlap: int) -> str:
    """Generate a sensible default output filename based on inputs."""
    base = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
    loader = "unstructured" if use_unstructured else "pypdf"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}__{loader}__cs{chunk_size}_co{chunk_overlap}__{timestamp}.json"

def main():
    print("PDF Parsing & Chunking Tool")
    print("Type 'exit' to quit.\n")

    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if pdf_path.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            use_unstructured = input("Use UnstructuredPDFLoader? (y/n): ").strip().lower() == "y"

            # Chunk params
            chunk_size_in = input("Enter chunk size (default 900): ").strip()
            chunk_overlap_in = input("Enter chunk overlap (default 180): ").strip()
            chunk_size = int(chunk_size_in) if chunk_size_in else 900
            chunk_overlap = int(chunk_overlap_in) if chunk_overlap_in else 180

            # Load
            print(f"\nLoading PDF: {pdf_path} (unstructured={use_unstructured})")
            docs = load_pdf(pdf_path, use_unstructured=use_unstructured)
            print(f"Loaded {len(docs)} document(s).")

            # Chunk
            print(f"Chunking with size={chunk_size}, overlap={chunk_overlap} ...")
            chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            num_chunks, min_len, max_len = summarize_chunks(chunks)
            print(f"Produced {num_chunks} chunks (min={min_len} chars, max={max_len} chars).")

            # Preview first few chunks
            preview_count_in = input("Preview how many chunks? (default 3): ").strip()
            preview_count = int(preview_count_in) if preview_count_in else 3
            preview_count = max(0, min(preview_count, num_chunks))

            for i, c in enumerate(chunks[:preview_count]):
                src = c.metadata.get("source", "unknown")
                page = c.metadata.get("page", "n/a")
                print("\n" + "-" * 80)
                print(f"Chunk {i+1} | source={src} | page={page} | length={len(c.page_content)}")
                print("-" * 80)
                print(c.page_content[:1000])  # up to 1000 chars

            # Ask to save
            save_choice = input("\nSave chunks to a JSON file? (y/n): ").strip().lower()
            if save_choice == "y":
                default_name = default_output_name(pdf_path, use_unstructured, chunk_size, chunk_overlap)
                out_path_in = input(f"Enter output filename (default '{default_name}'): ").strip()
                out_path = out_path_in or default_name

                # Ensure parent directory exists
                out_dir = os.path.dirname(out_path)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

                # Build payload
                payload = {
                    "source_pdf": os.path.abspath(pdf_path),
                    "loader": "UnstructuredPDFLoader" if use_unstructured else "PyPDFLoader",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_chunks": num_chunks,
                    "chunks": [document_to_dict(c) for c in chunks]
                }

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print(f"\n✅ Chunks saved to: {os.path.abspath(out_path)}")

            print("\nDone! You can now feed these chunks into embeddings later.\n")

        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()