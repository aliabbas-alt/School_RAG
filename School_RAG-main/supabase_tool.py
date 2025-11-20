# supabase_tool.py
"""
Enhanced semantic search tool for multimodal RAG with grade/subject filtering.
Features:
- Automatic grade/subject detection from queries
- Query enhancement using conversation memory
- Hybrid search (keyword + semantic) for examples
- Image description prioritization
- Smart result re-ranking
- Query type detection
- Better debugging and logging
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import Tool
from storage.supabase_storage import SupabaseVectorStore
from embeddings.registry import build_provider


class SmartSearchEngine:
    """Enhanced search engine with multimodal RAG and grade/subject filtering."""
    
    # def __init__(self):
    #     self.vector_store = SupabaseVectorStore()
    #     self.provider = build_provider("openai", model="text-embedding-3-small")
        
    #     # Search configuration
    #     self.default_threshold = 0.25
    #     self.default_limit = 30
    #     self.image_boost_factor = 1.2
    def __init__(self):
        self.vector_store = SupabaseVectorStore()
        #Search configuration
        self.default_threshold = 0.25
        self.default_limit = 30
        self.image_boost_factor = 1.2
    
    def enhance_query(self, query: str, memory_context: Optional[Dict] = None) -> str:
        """
        Automatically enhance vague queries with context from memory.
        
        Examples:
        - "example 8" + memory(integrals) → "integrals example 8"
        - "next example" + memory(Grade 12, Math) → "Grade 12 math next example"
        - "that diagram" + memory(page 23, Physics) → "physics diagram page 23"
        
        Args:
            query: Original user query
            memory_context: Dictionary with memory info (subjects, grades, pages, topics)
        
        Returns:
            Enhanced query with context added
        """
        if not memory_context:
            return query
        
        query_lower = query.lower()
        enhanced_parts = []
        
        # Extract what we know from memory
        subjects = memory_context.get('subjects_discussed', set())
        grades = memory_context.get('grades_discussed', set())
        pages = memory_context.get('pages_mentioned', set())
        last_topic = memory_context.get('last_topic', '')
        
        # Rule 1: If query mentions "example X" but no topic/subject
        if re.search(r'\bexample\s+\d+\b', query_lower):
            has_topic = any(word in query_lower for word in [
                'integral', 'derivative', 'matrix', 'equation', 
                'theorem', 'function', 'limit', 'series'
            ])
            
            if not has_topic and last_topic:
                enhanced_parts.append(last_topic)
                print(f"   💡 Enhanced: Added topic '{last_topic}' from memory")
        
        # Rule 2: If query says "next example/problem"
        if re.search(r'\b(next|another|show me|give me)\s+(example|problem|question)', query_lower):
            if last_topic:
                enhanced_parts.append(last_topic)
                print(f"   💡 Enhanced: Added topic '{last_topic}' for 'next' query")
        
        # Rule 3: Pronouns referring to previous context
        if re.search(r'\b(that|this|the)\s+(diagram|figure|image|example|problem)', query_lower):
            if subjects:
                subject = list(subjects)  # FIX: was list(subjects) without
                enhanced_parts.append(subject)
                print(f"   💡 Enhanced: Added subject '{subject}' for pronoun reference")
        
        # Rule 4: Vague "explain it" type queries
        if query_lower in ['explain', 'show me', 'tell me more', 'continue', 'go on']:
            if last_topic:
                enhanced_parts.append(last_topic)
            if subjects:
                enhanced_parts.append(list(subjects))  # FIX: was list(subjects) without
            print(f"   💡 Enhanced: Added context for vague query")
        
        # Rule 5: Add grade if asking about general topics and grade is known
        if grades and not re.search(r'\bgrade\s+\d+\b', query_lower):
            if any(word in query_lower for word in ['chapter', 'textbook', 'curriculum', 'syllabus']):
                grade = list(grades)  # FIX: was list(grades) without
                enhanced_parts.append(f"grade {grade}")
                print(f"   💡 Enhanced: Added grade {grade} context")
        
        # Combine enhanced parts with original query
        if enhanced_parts:
            enhanced_query = ' '.join(enhanced_parts) + ' ' + query
            return enhanced_query
        
        return query
    
    def detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect query intent and extract parameters including grade/subject."""
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "is_image_query": False,
            "is_page_specific": False,
            "is_example_query": False,
            "example_number": None,
            "pages": [],
            "keywords": [],
            "grade": None,
            "subject": None
        }
        
        # Check if asking for specific example
        example_match = re.search(r'example\s+(\d+)', query_lower)
        if example_match:
            intent["is_example_query"] = True
            intent["example_number"] = int(example_match.group(1))
            intent["type"] = "example"
        
        # Check if image-related query
        image_keywords = ["image", "diagram", "figure", "picture", "illustration", 
                         "photo", "drawing", "chart", "graph", "visual"]
        if any(kw in query_lower for kw in image_keywords):
            intent["is_image_query"] = True
            intent["type"] = "visual"
        
        # Extract page numbers
        page_pattern = r'\bpage\s+(\d+)\b|\bp\.?\s*(\d+)\b'
        pages = re.findall(page_pattern, query_lower)
        if pages:
            intent["is_page_specific"] = True
            # pages is list of (g1, g2); pick the non-empty group and convert to int
            intent["pages"] = [int(a or b) for (a, b) in pages]
            if intent.get("type") == "general":
                intent["type"] = "page_specific"
        
        # Extract grade level
        grade_pattern = r'\bgrade\s+(\d+)\b|\bclass\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+grade\b'
        grade_matches = re.findall(grade_pattern, query_lower)
        if grade_matches:
            for match in grade_matches:
                # match is a tuple like (g1, g2, g3); pick first non-empty group
                num_str = next((g for g in match if g), None)
                if not num_str:
                    continue
                grade_num = int(num_str)
                if 1 <= grade_num <= 12:
                    intent["grade"] = grade_num
                    break
        
        # Extract subject with better topic detection
        subjects_map = {
            "Math": ["math", "mathematics", "algebra", "geometry", "calculus", "arithmetic",
                    "integral", "derivative", "equation", "function", "theorem", "trigonometry"],
            "Physics": ["physics", "mechanics", "thermodynamics", "optics", "circuit", "force"],
            "Chemistry": ["chemistry", "chemical", "organic", "inorganic", "molecule", "reaction"],
            "Biology": ["biology", "bio", "botany", "zoology", "life science", "cell", "organism"],
            "English": ["english", "language", "literature", "grammar", "essay", "poetry"],
            "History": ["history", "historical"],
            "Geography": ["geography", "geo"],
            "Computer": ["computer", "programming", "coding", "technology"],
            "Science": ["science", "scientific"]
        }
        
        for subject_key, keywords in subjects_map.items():
            if any(kw in query_lower for kw in keywords):
                intent["subject"] = subject_key
                break
        
        # Extract important keywords
        important_words = [
            word for word in query_lower.split() 
            if len(word) > 3 and word not in ["what", "where", "when", "which", "describe", "explain", "show"]
        ]
        intent["keywords"] = important_words[:5]
        
        return intent
    
    def keyword_search_examples(self, example_num: int, subject: str = None) -> List[Dict]:
        """
        Direct keyword search for specific examples.
        Prioritizes math_image (vision) over math_text (OCR).
        
        Args:
            example_num: Example number to search for
            subject: Optional subject filter (e.g., "Math")
        
        Returns:
            List of matching documents, vision-extracted first
        """
        try:
            # Build query
            # Subject-aware table selection
            table_name = "math_documents" if subject and subject.lower() == "math" else "documents"

            query = (
                self.vector_store.client
                .table(table_name)
                .select("*")
                .ilike("content", f"%Example {example_num}%")
)
            
            # Add subject filter if provided
            if subject:
                query = query.eq("subject", subject)
            
            # Execute
            response = query.limit(20).execute()
            
            if not (hasattr(response, 'data') and response.data):
                return []
            
            # Manual sorting: math_image first, then by page
            results = response.data
            
            # Separate by type
            vision_results = [r for r in results if r.get('content_type') == 'math_image']
            text_results = [r for r in results if r.get('content_type') == 'math_text']
            other_results = [r for r in results if r.get('content_type') not in ['math_image', 'math_text']]
            
            # Sort each group by page
            vision_results.sort(key=lambda x: x.get('page', 999))
            text_results.sort(key=lambda x: x.get('page', 999))
            other_results.sort(key=lambda x: x.get('page', 999))
            
            # Combine: vision first
            all_results = vision_results + text_results + other_results
            
            # Add synthetic similarity scores
            for i, doc in enumerate(all_results):
                doc['similarity'] = 1.0 - (i * 0.03)
            
            return all_results[:10]
        
        except Exception as e:
            print(f"   ⚠️ Keyword search error: {e}")
            return []
    
    def search_with_intent(
        self, 
        query: str, 
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search with intent-aware configuration and grade/subject filters."""
        
        # NEW: For example queries, try keyword search first
        if intent.get("is_example_query"):
            example_num = intent["example_number"]
            subject = intent.get("subject")
            
            print(f"   🎯 Trying keyword search for Example {example_num}...")
            keyword_results = self.keyword_search_examples(example_num, subject)
            
            if keyword_results:
                print(f"   ✅ Found {len(keyword_results)} results via keyword search")
                return keyword_results
            else:
                print(f"   ⚠️ No keyword results, falling back to semantic search")
        
        # Original semantic search

        # NEW: Pick provider dynamically based on subject
        subject = intent.get("subject")
        if subject and subject.lower() == "math":
            provider = build_provider("openai", model="text-embedding-3-large")
            print("   🔮 Using text-embedding-3-large for Math query")
        else:
            provider = build_provider("openai", model="text-embedding-3-small")
            print("   🔮 Using text-embedding-3-small for non-Math query")

        query_result = provider.embed_texts([query])
        query_embedding = query_result.vectors
        
        # Adjust search parameters based on intent
        if intent["is_image_query"]:
            threshold = 0.20
            limit = 40
        elif intent["is_page_specific"]:
            threshold = 0.30
            limit = 20
        elif intent["is_example_query"]:
            threshold = 0.28
            limit = 25
        else:
            threshold = self.default_threshold
            limit = self.default_limit
        
        # Perform search with grade/subject filters
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            match_threshold=threshold,
            match_count=limit,
            school_id=None,
            curriculum_type=None,
            document_type=None,
            academic_year=None,
            grade=intent.get("grade"),
            subject=intent.get("subject")
        )
        
        return results
    
    def rerank_results(
        self, 
        results: List[Dict[str, Any]], 
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on query intent and content type.
        Prioritizes vision-extracted math over OCR text.
        """
        if not results:
            return results
        
        # Separate content by type and quality
        vision_math = []
        text_math = []
        image_desc = []
        other = []
        
        for result in results:
            content_type = result.get('content_type', 'text')
            
            if content_type == 'math_image':
                result['similarity'] = min(1.0, result.get('similarity', 0) * 2.0)
                vision_math.append(result)
            elif content_type == 'math_text':
                text_math.append(result)
            elif content_type == 'image_description':
                image_desc.append(result)
            else:
                other.append(result)
        
        # Simple re-ranking based on intent
        if intent.get("is_example_query"):
            return vision_math + text_math + image_desc + other
        
        elif intent.get("is_image_query"):
            for img in image_desc:
                img['similarity'] = min(1.0, img.get('similarity', 0) * 1.2)
            return vision_math + image_desc + text_math + other
        
        elif intent.get("is_page_specific"):
            target_pages = set(intent.get("pages", []))
            all_results = vision_math + text_math + image_desc + other
            
            exact = [r for r in all_results if r.get('page') in target_pages]
            partial = [r for r in all_results if r.get('page') not in target_pages]
            return exact + partial
        
        else:
            return vision_math + text_math + image_desc + other
    
    def format_results(
        self, 
        results: List[Dict[str, Any]], 
        max_results: int = 10,
        verbose: bool = False
    ) -> str:
        """Format search results for LLM consumption with grade/subject metadata."""
        if not results:
            return "No relevant documents found in the database."
        
        formatted_parts = []
        
        if verbose:
            image_count = sum(1 for r in results if r.get('content_type') == 'image_description')
            text_count = len(results) - image_count
            
            grades = set(r.get('grade') for r in results if r.get('grade'))
            subjects = set(r.get('subject') for r in results if r.get('subject'))
            
            formatted_parts.append("="*70)
            formatted_parts.append(f"Search Results: {len(results)} total ({image_count} images, {text_count} text)")
            if grades:
                formatted_parts.append(f"Grades: {', '.join(map(str, sorted(grades)))}")
            if subjects:
                formatted_parts.append(f"Subjects: {', '.join(subjects)}")
            formatted_parts.append("="*70)
            formatted_parts.append("")
        
        # Format top results
        for i, result in enumerate(results[:max_results], 1):
            source = result.get('source', 'Unknown')
            
            if '\\' in source:
                source = source.split('\\')[-1]
            elif '/' in source:
                source = source.split('/')[-1]
            
            page = result.get('page', 'N/A')
            content = result.get('content', '')
            content_type = result.get('content_type', 'text')
            similarity = result.get('similarity', 0)
            
            grade = result.get('grade')
            subject = result.get('subject')
            
            type_label = "IMAGE DESCRIPTION" if content_type == 'image_description' else "TEXT"
            
            metadata_parts = []
            if grade:
                metadata_parts.append(f"Grade {grade}")
            if subject:
                metadata_parts.append(subject)
            
            metadata_str = f" | {' | '.join(metadata_parts)}" if metadata_parts else ""
            
            formatted_parts.append(f"【Result {i}】 {type_label}{metadata_str} | Page {page} | Match: {similarity:.0%}")
            formatted_parts.append(f"Source: {source}")
            formatted_parts.append("-" * 70)
            
            if content_type == 'image_description':
                formatted_parts.append(content)
            else:
                if len(content) > 1000:
                    formatted_parts.append(content[:1000] + "\n... [truncated]")
                else:
                    formatted_parts.append(content)
            
            formatted_parts.append("")
            formatted_parts.append("="*70)
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def search(self, query: str, memory_context: Optional[Dict] = None, verbose: bool = False) -> str:
        """
        Main search function with query enhancement and all features.
        
        Args:
            query: User query
            memory_context: Dictionary with memory info (subjects, grades, pages, topics)
            verbose: Enable debug output
        
        Returns:
            Formatted search results
        """
        # Step 0: Enhance query with memory context
        enhanced_query = self.enhance_query(query, memory_context)
        
        if enhanced_query != query and verbose:
            print(f"   📝 Original: {query}")
            print(f"   ✨ Enhanced: {enhanced_query}")
        
        # Step 1: Detect intent
        intent = self.detect_query_intent(enhanced_query)
        
        if verbose:
            print(f"\n🔍 Query Intent: {intent['type']}")
            if intent['is_image_query']:
                print("   → Image-focused search enabled")
            if intent['is_page_specific']:
                print(f"   → Page-specific: {intent['pages']}")
            if intent['is_example_query']:
                print(f"   → Example search: {intent['example_number']}")
            if intent.get('grade'):
                print(f"   → Grade filter: {intent['grade']}")
            if intent.get('subject'):
                print(f"   → Subject filter: {intent['subject']}")
        
        # Step 2: Search with filters
        results = self.search_with_intent(enhanced_query, intent)
        
        if verbose:
            print(f"   → Found {len(results)} initial results")
        
        # Step 3: Re-rank
        reranked_results = self.rerank_results(results, intent)
        
        if verbose:
            image_count = sum(1 for r in reranked_results if r.get('content_type') == 'image_description')
            print(f"   → After re-ranking: {image_count} image descriptions prioritized")
        
        # Step 4: Format
        formatted = self.format_results(reranked_results, max_results=10, verbose=verbose)
        
        return formatted


# Global search engine instance
_search_engine = None


def get_search_engine() -> SmartSearchEngine:
    """Get or create search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SmartSearchEngine()
    return _search_engine


def search_school_documents(query: str, memory_context: Optional[Dict] = None) -> str:
    """
    Enhanced search function with memory-based query enhancement and hybrid search.
    
    Args:
        query: User's search query
        memory_context: Optional dict with memory info (subjects, grades, pages, last_topic)
    
    Returns:
        Formatted search results
    """
    try:
        engine = get_search_engine()
        return engine.search(query, memory_context=memory_context, verbose=True)
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Search Error: {str(e)}\n\n"
        error_msg += "Traceback:\n" + traceback.format_exc()
        return error_msg


# Create LangChain Tool
supabase_tool = Tool.from_function(
    name="search_school_documents",
    description="""Search comprehensive educational database with hybrid search (keyword + semantic), automatic grade/subject filtering, and memory-enhanced query understanding.

**Database Organization:**
- Content organized by GRADE (1-12) and SUBJECT
- Complete textbook content (text + AI-generated image descriptions)
- School policies, handbooks, procedures
- Curriculum materials (CBSE, SSE, ICSE, IB, etc.)

**Hybrid Search:**
- Example queries use direct keyword matching for 100% accuracy
- Falls back to semantic search if keyword fails
- Prioritizes vision-extracted math over OCR text

**Smart Query Enhancement:**
The system uses conversation memory to enhance vague queries:
- "example 8" → automatically adds topic from memory (e.g., "integrals example 8")
- "next example" → continues from previous context
- "that diagram" → knows which subject/page you're referring to

**Automatic Filtering:**
Detects and filters by:
- Grade level from queries
- Subject from keywords
- Page numbers
- Content type (images vs text)
- Example numbers

**Use for:**
- "Example 8" (uses keyword search + memory context)
- "Grade 6 English images on page 15"
- "Show me the next problem"
- "Explain that diagram" (context-aware)
- Any textbook content questions

Returns detailed results with source citations and grade/subject metadata.""",
    func=search_school_documents,
    return_direct=False
)
