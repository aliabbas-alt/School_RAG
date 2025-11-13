# math_parser.py
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF
from sympy import sympify, latex
from ingest.pdf_loader import load_pdf
from ingest.chunk_utils import chunk_documents
from openai import OpenAI
import re
import base64
 
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
def normalize_math_text(text: str) -> str:
    """
    Normalize common OCR quirks in math text before SymPy conversion.
    """
    # --- Superscripts ---
    text = text.replace("²", "**2").replace("³", "**3")
    text = text.replace("^", "**")  # caret to SymPy exponent
 
    # --- Minus signs / dashes ---
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
 
    # --- Differential spacing ---
    text = re.sub(r"d\s+x", "dx", text)   # collapse "d x"
    text = text.replace("d×", "dx")
 
    # --- Function names ---
    text = re.sub(r"\bs\s*in\b", "sin", text)
    text = re.sub(r"\bc\s*os\b", "cos", text)
    text = re.sub(r"\bt\s*an\b", "tan", text)
    text = re.sub(r"\bl\s*og\b", "log", text)
    text = text.replace("1og", "log")  # OCR '1' vs 'l'
 
    # --- Greek letters ---
    text = text.replace("π", "pi").replace("Π", "pi")
    text = text.replace("θ", "theta").replace("Θ", "theta")
 
    # --- Fraction bar loss (basic heuristic) ---
    text = re.sub(r"(\d)([a-zA-Z])", r"\1/\2", text)
 
    return text
 
def convert_text_to_latex(text: str) -> str:
    try:
        normalized = normalize_math_text(text)
        expr = sympify(normalized)
        return f"${latex(expr)}$"
    except Exception:
        return f"${text}$"
 
def convert_image_to_latex(image_bytes: bytes) -> str:
    """
    Use GPT-4o mini vision to extract LaTeX from images/diagrams.
    ✅ FIXED VERSION with correct API format.
    """
    try:
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        # ✅ CORRECT API FORMAT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a math assistant. Extract all mathematical equations, formulas, and diagrams from the image and return them in LaTeX format. Be detailed and precise."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all mathematical content from this image and convert it to LaTeX. Include equations, formulas, and describe any diagrams or graphs."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,  # Increased for detailed descriptions
            temperature=0
        )
        latex_content = response.choices[0].message.content.strip()
        # Validate we got something useful
        if len(latex_content) < 10:
            return "No mathematical content detected in image"
        return latex_content
    except Exception as e:
        return f"Image-to-LaTeX failed: {str(e)}"
 
def process_math_pdf(pdf_path: str, use_unstructured: bool, chunk_size: int, chunk_overlap: int):
    """Process PDF text content and convert to LaTeX."""
    docs = load_pdf(pdf_path, use_unstructured=use_unstructured)
    text_chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 
    math_chunks = []
    for c in text_chunks:
        math_chunks.append({
            "page_content": convert_text_to_latex(c.page_content),
            "metadata": {
                **c.metadata,
                "content_type": "math_text",
                "math_format": "latex"
            }
        })
    return math_chunks
 
def process_math_images(pdf_path: str, output_dir: str = "extracted_images", dpi: int = 200):
    """
    Extract images from PDF pages using PyMuPDF and convert them to LaTeX with GPT-4o mini.
    ✅ Now uses correct API format.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return []
 
    vision_chunks = []
    pdf_name = Path(pdf_path).stem
    total_pages = len(pdf_document)
 
    print(f"\n🔍 Processing {total_pages} pages for images...")
 
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
 
        image_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num + 1}.png")
        pix.save(image_path)
 
        print(f"   Processing page {page_num + 1}/{total_pages}...", end=" ")
 
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            latex_equation = convert_image_to_latex(image_bytes)
            print(f"✅ Extracted {len(latex_equation)} chars")
        except Exception as e:
            latex_equation = f"❌ Image-to-LaTeX failed: {str(e)}"
            print(f"❌ Error: {e}")
 
        vision_chunks.append({
            "page_content": latex_equation,
            "metadata": {
                "page": page_num + 1,
                "pdf_source": os.path.basename(pdf_path),
                "image_path": image_path,
                "content_type": "math_image",
                "math_format": "latex",
                "source_type": "gpt4o_mini_vision"
            }
        })
 
    pdf_document.close()
    print(f"\n✅ Processed {len(vision_chunks)} image pages")
    return vision_chunks
 
def default_output_name(pdf_path: str) -> str:
    """Generate default output filename."""
    base = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}__math__{timestamp}.json"
 
def main():
    print("="*70)
    print("📘 Math PDF Parser (SymPy for text + GPT-4o mini for images)")
    print("="*70)
    pdf_path = input("\n Enter the path to your math PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    use_unstructured = False
    chunk_size, chunk_overlap = 900, 180
 
    print("\n📄 Processing text content...")
    math_chunks = process_math_pdf(pdf_path, use_unstructured, chunk_size, chunk_overlap)
    print(f"✅ Created {len(math_chunks)} text chunks")
 
    print("\n🖼️  Processing images...")
    vision_chunks = process_math_images(pdf_path)
 
    all_chunks = math_chunks + vision_chunks
 
    output_name = default_output_name(pdf_path)
    with open(output_name, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, ensure_ascii=False, indent=2)
 
    print(f"\n{'='*70}")
    print(f"✅ Saved {len(all_chunks)} total chunks to {output_name}")
    print(f"   - {len(math_chunks)} text chunks")
    print(f"   - {len(vision_chunks)} image chunks")
    print(f"{'='*70}")
 
    # 🔗 Call Supabase storage script
    upload = input("\n📥 Upload to Supabase now? (y/n): ").strip().lower()
    if upload == 'y':
        from storage import store_to_supabase
        store_to_supabase.main()
 
if __name__ == "__main__":
    main()