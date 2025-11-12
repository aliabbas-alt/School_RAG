# supabase_tool.py
"""
Enhanced semantic search tool for multimodal RAG with grade/subject filtering.
Features:
- Automatic grade/subject detection from queries
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
    
    def __init__(self):
        self.vector_store = SupabaseVectorStore()
        self.provider = build_provider("openai", model="text-embedding-3-small")
        
        # Search configuration
        self.default_threshold = 0.25
        self.default_limit = 30
        self.image_boost_factor = 1.2  # Boost image results
    
    def detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect query intent and extract parameters including grade/subject.
        
        Returns:
            Dictionary with query intent, filters, and metadata
        """
        query_lower = query.lower()
        
        intent = {
            "type": "general",
            "is_image_query": False,
            "is_page_specific": False,
            "pages": [],
            "keywords": [],
            "grade": None,
            "subject": None
        }
        
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
            intent["pages"] = [int(m[0] or m[1]) for m in pages]
            intent["type"] = "page_specific"
        
        # Extract grade level
        grade_pattern = r'\bgrade\s+(\d+)\b|\bclass\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+grade\b'
        grade_matches = re.findall(grade_pattern, query_lower)
        if grade_matches:
            for match in grade_matches:
                grade_num = int(match[0] or match[1] or match[2])
                if 1 <= grade_num <= 12:
                    intent["grade"] = grade_num
                    break
        
        # Extract subject
        subjects_map = {
            "Math": ["math", "mathematics", "algebra", "geometry", "calculus", "arithmetic"],
            "Physics": ["physics", "mechanics", "thermodynamics", "optics"],
            "Chemistry": ["chemistry", "chemical", "organic", "inorganic"],
            "Biology": ["biology", "bio", "botany", "zoology", "life science"],
            "English": ["english", "language", "literature", "grammar"],
            "History": ["history", "historical"],
            "Geography": ["geography", "geo"],
            "Computer": ["computer", "programming", "coding", "technology"],
            "Science": ["science", "scientific"]  # General science
        }
        
        for subject_key, keywords in subjects_map.items():
            if any(kw in query_lower for kw in keywords):
                intent["subject"] = subject_key
                break
        
        # Extract important keywords
        important_words = [
            word for word in query_lower.split() 
            if len(word) > 3 and word not in ["what", "where", "when", "which", "describe", "explain"]
        ]
        intent["keywords"] = important_words[:5]
        
        return intent
    
    def search_with_intent(
        self, 
        query: str, 
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search with intent-aware configuration and grade/subject filters.
        
        Args:
            query: Search query
            intent: Query intent from detect_query_intent()
        
        Returns:
            Search results
        """
        # Generate query embedding
        query_result = self.provider.embed_texts([query])
        query_embedding = query_result.vectors[0]
        
        # Adjust search parameters based on intent
        if intent["is_image_query"]:
            # For image queries, cast a wider net
            threshold = 0.20
            limit = 40
        elif intent["is_page_specific"]:
            # For page-specific queries, be more precise
            threshold = 0.30
            limit = 20
        else:
            # General queries
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
        
        Args:
            results: Raw search results
            intent: Query intent
        
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        # Separate by content type
        image_results = []
        text_results = []
        
        for result in results:
            content_type = result.get('content_type', 'text')
            if content_type == 'image_description':
                image_results.append(result)
            else:
                text_results.append(result)
        
        # Re-rank based on intent
        if intent["is_image_query"]:
            # Prioritize image descriptions for image queries
            # Boost their scores
            for img_result in image_results:
                img_result['similarity'] = min(1.0, img_result.get('similarity', 0) * self.image_boost_factor)
            
            # Combine: images first, then text
            reranked = image_results + text_results
        
        elif intent["is_page_specific"]:
            # For page-specific queries, filter and prioritize exact page matches
            target_pages = set(intent["pages"])
            exact_matches = []
            partial_matches = []
            
            for result in results:
                page = result.get('page')
                if page in target_pages:
                    exact_matches.append(result)
                else:
                    partial_matches.append(result)
            
            reranked = exact_matches + partial_matches
        
        else:
            # General queries: mixed results, slight preference for images
            # Sort by similarity, but boost images slightly
            scored_results = []
            for result in results:
                score = result.get('similarity', 0)
                if result.get('content_type') == 'image_description':
                    score *= 1.1  # Small boost
                scored_results.append((score, result))
            
            scored_results.sort(reverse=True, key=lambda x: x[0])
            reranked = [r for _, r in scored_results]
        
        return reranked
    
    def format_results(
        self, 
        results: List[Dict[str, Any]], 
        max_results: int = 10,
        verbose: bool = False
    ) -> str:
        """
        Format search results for LLM consumption with grade/subject metadata.
        
        Args:
            results: Search results
            max_results: Maximum results to include
            verbose: Include debug information
        
        Returns:
            Formatted results string
        """
        if not results:
            return "No relevant documents found in the database."
        
        formatted_parts = []
        
        if verbose:
            image_count = sum(1 for r in results if r.get('content_type') == 'image_description')
            text_count = len(results) - image_count
            
            # Show grade/subject distribution
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
            
            # Extract filename only
            if '\\' in source:
                source = source.split('\\')[-1]
            elif '/' in source:
                source = source.split('/')[-1]
            
            page = result.get('page', 'N/A')
            content = result.get('content', '')
            content_type = result.get('content_type', 'text')
            similarity = result.get('similarity', 0)
            
            # Get grade and subject
            grade = result.get('grade')
            subject = result.get('subject')
            
            # Format header with grade/subject
            type_label = "IMAGE DESCRIPTION" if content_type == 'image_description' else "TEXT"
            
            # Build metadata line
            metadata_parts = []
            if grade:
                metadata_parts.append(f"Grade {grade}")
            if subject:
                metadata_parts.append(subject)
            
            metadata_str = f" | {' | '.join(metadata_parts)}" if metadata_parts else ""
            
            formatted_parts.append(f"【Result {i}】 {type_label}{metadata_str} | Page {page} | Match: {similarity:.0%}")
            formatted_parts.append(f"Source: {source}")
            formatted_parts.append("-" * 70)
            
            # Format content
            # CRITICAL: Show FULL content for image descriptions
            if content_type == 'image_description':
                formatted_parts.append(content)  # NO TRUNCATION
            else:
                # Truncate text results
                if len(content) > 1000:
                    formatted_parts.append(content[:1000] + "\n... [truncated]")
                else:
                    formatted_parts.append(content)
            
            formatted_parts.append("")
            formatted_parts.append("="*70)
            formatted_parts.append("")
        
        return "\n".join(formatted_parts)
    
    def search(self, query: str, verbose: bool = False) -> str:
        """
        Main search function with all enhancements including grade/subject.
        
        Args:
            query: User query
            verbose: Enable debug output
        
        Returns:
            Formatted search results
        """
        # Step 1: Detect intent (includes grade/subject)
        intent = self.detect_query_intent(query)
        
        if verbose:
            print(f"\n🔍 Query Intent: {intent['type']}")
            if intent['is_image_query']:
                print("   → Image-focused search enabled")
            if intent['is_page_specific']:
                print(f"   → Page-specific: {intent['pages']}")
            if intent.get('grade'):
                print(f"   → Grade filter: {intent['grade']}")
            if intent.get('subject'):
                print(f"   → Subject filter: {intent['subject']}")
        
        # Step 2: Search with filters
        results = self.search_with_intent(query, intent)
        
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


def search_school_documents(query: str) -> str:
    """
    Enhanced search function for LangChain tool with grade/subject support.
    
    Args:
        query: User's search query
    
    Returns:
        Formatted search results with automatic grade/subject filtering
    """
    try:
        engine = get_search_engine()
        return engine.search(query, verbose=True)
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Search Error: {str(e)}\n\n"
        error_msg += "Traceback:\n" + traceback.format_exc()
        return error_msg


# Create LangChain Tool
supabase_tool = Tool.from_function(
    name="search_school_documents",
    description="""Search comprehensive educational database with advanced multimodal RAG and automatic grade/subject filtering.

**Database Organization:**
- Content organized by GRADE (1-12) and SUBJECT
- Complete textbook content (text + AI-generated image descriptions)
- School policies, handbooks, procedures
- Curriculum materials (CBSE, SSE, ICSE, IB, etc.)

**Automatic Filtering:**
The search automatically detects and filters by:
- Grade level from queries (e.g., "Grade 8 math problem")
- Subject from keywords (Math, Physics, Chemistry, Biology, English, etc.)
- Page numbers (e.g., "page 16")
- Content type (images vs text)

**Capabilities:**
- Text search with semantic understanding
- Image/diagram descriptions (complete, with colors and positions)
- Visual element search (figures, charts, activities)
- Grade-appropriate content filtering
- Subject-specific searches
- Multi-document search across grades/subjects

**Special Features:**
- Automatically prioritizes image descriptions for visual queries
- Re-ranks results based on query intent
- Provides complete, untruncated image descriptions
- Smart grade/subject detection from natural language

**Use for:**
- "Grade 6 English images on page 15"
- "Physics Grade 10 circuit diagrams"
- "Math Grade 8 chapter 3 exercises"
- "What diagrams show photosynthesis?" (auto-detects Biology/Science)
- "Chemistry Grade 11 chemical reactions"
- Any textbook content questions with automatic filtering

Returns detailed, properly formatted results with source citations and grade/subject metadata.""",
    func=search_school_documents,
    return_direct=False
)
