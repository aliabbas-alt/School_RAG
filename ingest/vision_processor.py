# ingest/vision_processor.py (PyMuPDF version - NO POPPLER NEEDED)
import os
import base64
from typing import List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import io

load_dotenv()


class VisionProcessor:
    """Extract and describe images from PDFs using PyMuPDF (no Poppler needed)."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.vision_model = ChatOpenAI(
            model="gpt-4o",
            api_key=self.api_key,
            temperature=0.1,
            max_tokens=2000
        )
    
    def extract_images_from_pdf(
        self, 
        pdf_path: str, 
        output_dir: str = "extracted_images",
        dpi: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Extract images using PyMuPDF (no Poppler dependency).
        """
        print(f"\n📄 Extracting images from: {os.path.basename(pdf_path)}")
        print(f"   Using PyMuPDF (no Poppler required)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return []
        
        image_data = []
        pdf_name = Path(pdf_path).stem
        
        # Convert each page to image
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Render page to image (matrix for DPI scaling)
            zoom = dpi / 72  # 72 is default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            image_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num + 1}.png")
            pix.save(image_path)
            
            image_data.append({
                "image_path": image_path,
                "page_number": page_num + 1,
                "pdf_source": os.path.basename(pdf_path),
                "file_size_mb": os.path.getsize(image_path) / (1024 * 1024)
            })
            
            print(f"   ✅ Page {page_num + 1}: {image_path} ({image_data[-1]['file_size_mb']:.1f} MB)")
        
        pdf_document.close()
        print(f"\n✅ Extracted {len(image_data)} pages\n")
        return image_data
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_detailed_description(
        self, 
        image_path: str, 
        page_number: int,
        document_context: str = "educational textbook"
    ) -> str:
        """Generate ultra-detailed description using GPT-4 Vision."""
        print(f"🔍 Analyzing Page {page_number} with GPT-4 Vision...")
        
        base64_image = self.encode_image(image_path)
        
        prompt = f"""You are analyzing Page {page_number} of a {document_context}.

CRITICAL: Provide an EXTREMELY DETAILED description that captures EVERY element. DO NOT LEAVE ANYTHING OUT.

Your description must include:

1. **ALL TEXT CONTENT**
   - Transcribe ALL headings, titles, and headers (exact wording)
   - Transcribe ALL body text, paragraphs, and captions (verbatim)
   - Transcribe ALL labels, numbers, and annotations
   - Transcribe ALL questions, instructions, and activity descriptions

2. **VISUAL ELEMENTS**
   - Describe every image, illustration, photo, or diagram
   - Explain what each visual element depicts
   - Describe colors, positions, and spatial relationships
   - Identify figure numbers (Fig. 1, Fig. 2, etc.)

3. **EDUCATIONAL COMPONENTS**
   - Identify activities, exercises, or questions
   - Describe any tables, charts, or graphs
   - Note any "Try Again", "Think About It" sections
   - Describe hands-on activities or experiments

4. **SPECIFIC DETAILS**
   - Count and number all images
   - Describe each numbered figure individually
   - Include page numbers or chapter markers

Format as comprehensive description that someone could use to fully understand the page."""

        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            response = self.vision_model.invoke([message])
            description = response.content
            
            full_description = f"""=== PAGE {page_number} VISUAL CONTENT ===

{description}

[End of Page {page_number} description]"""
            
            print(f"   ✅ Generated {len(full_description)} character description")
            return full_description
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return f"[Error analyzing Page {page_number}: {str(e)}]"
    
    def process_pdf_with_vision(
        self,
        pdf_path: str,
        output_dir: str = "extracted_images",
        document_type: str = "educational textbook"
    ) -> List[Dict[str, Any]]:
        """Complete pipeline: Extract → Describe."""
        print("="*70)
        print("VISION-POWERED PDF PROCESSING (PyMuPDF - No Poppler)")
        print("="*70)
        
        image_data = self.extract_images_from_pdf(pdf_path, output_dir)
        
        if not image_data:
            return []
        
        print("="*70)
        print("GENERATING DETAILED DESCRIPTIONS")
        print("="*70)
        
        for idx, img_info in enumerate(image_data, 1):
            print(f"\nProcessing {idx}/{len(image_data)}...")
            
            description = self.generate_detailed_description(
                img_info["image_path"],
                img_info["page_number"],
                document_type
            )
            
            img_info["description"] = description
            img_info["description_length"] = len(description)
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*70}\n")
        
        return image_data
