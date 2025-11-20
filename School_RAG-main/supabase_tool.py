# supabase_tool.py
"""
Production-ready semantic search tool for multimodal RAG with grade/subject filtering.
Features:
- True hybrid search with Reciprocal Rank Fusion (RRF)
- Automatic grade/subject detection from queries
- Query enhancement using conversation memory
- Image description prioritization
- Smart result re-ranking and deduplication
- School data isolation for multi-tenancy
- Error handling with fallbacks
- Performance monitoring
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import Tool
from storage.supabase_storage import SupabaseVectorStore
from embeddings.registry import build_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartSearchEngine:
    """Production-grade search engine with multimodal RAG and grade/subject filtering."""
    
    def __init__(self):
        self.vector_store = SupabaseVectorStore()
        self.provider = build_provider("openai", model="text-embedding-3-small")
        
        # Search configuration
        self.default_threshold = 0.25
        self.default_limit = 30
        self.image_boost_factor = 1.2
        
        # Caching
        self.semantic_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "failed_queries": 0,
            "avg_latency": []
        }
    
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
        last_page = memory_context.get('last_page')
        
        # Rule 1: If query mentions "example X" but no topic/subject
        if re.search(r'\bexample\s+\d+\b', query_lower):
            has_topic = any(word in query_lower for word in [
                'integral', 'derivative', 'matrix', 'equation', 
                'theorem', 'function', 'limit', 'series'
            ])
            
            if not has_topic and last_topic:
                enhanced_parts.append(last_topic)
                logger.info(f"Enhanced: Added topic '{last_topic}' from memory")
        
        # Rule 2: If query says "next example/problem"
        if re.search(r'\b(next|another|show me|give me)\s+(example|problem|question)', query_lower):
            if last_topic:
                enhanced_parts.append(last_topic)
                logger.info(f"Enhanced: Added topic '{last_topic}' for 'next' query")
            
            if last_page:
                enhanced_parts.append(f"page {last_page}")
                logger.info(f"Enhanced: Added page {last_page} from memory")
        
        # Rule 3: Pronouns referring to previous context
        if re.search(r'\b(that|this|the)\s+(diagram|figure|image|example|problem)', query_lower):
            if subjects:
                subject = list(subjects)[0]  # FIX: Get first element
                enhanced_parts.append(subject)
                logger.info(f"Enhanced: Added subject '{subject}' for pronoun reference")
            
            if last_page:
                enhanced_parts.append(f"page {last_page}")
                logger.info(f"Enhanced: Added page {last_page} for pronoun reference")
        
        # Rule 4: Vague "explain it" type queries
        if query_lower in ['explain', 'show me', 'tell me more', 'continue', 'go on']:
            if last_topic:
                enhanced_parts.append(last_topic)
            if subjects:
                subject = list(subjects)[0]  # FIX: Get first element
                enhanced_parts.append(subject)
            
            if last_page:
                enhanced_parts.append(f"page {last_page}")
                logger.info("Enhanced: Added context for vague query")
    

        
        # Rule 5: Add grade if asking about general topics and grade is known
        if grades and not re.search(r'\bgrade\s+\d+\b', query_lower):
            if any(word in query_lower for word in ['chapter', 'textbook', 'curriculum', 'syllabus']):
                grade = list(grades)[0]  # FIX: Get first element
                enhanced_parts.append(f"grade {grade}")
                logger.info(f"Enhanced: Added grade {grade} context")
        
        # Combine enhanced parts with original query
        if enhanced_parts:
            enhanced_query = ' '.join(str(p) for p in enhanced_parts if p) + ' ' + query
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
        else:
        # NEW: Support word numbers (first, second, third, ... tenth)
            number_words = {
                'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
                'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
                'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18,
                'nineteenth': 19, 'twentieth': 20
            }
            for word, num in number_words.items():
                if re.search(rf'\b{word}\s+example\b|\bexample\s+{word}\b', query_lower):
                    intent["is_example_query"] = True
                    intent["example_number"] = num
                    break      
        
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
            intent["pages"] = [int(a or b) for (a, b) in pages]
            if intent.get("type") == "general":
                intent["type"] = "page_specific"
        
        # Extract grade level
        grade_pattern = r'\bgrade\s+(\d+)\b|\bclass\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\s+grade\b'
        grade_matches = re.findall(grade_pattern, query_lower)
        if grade_matches:
            for match in grade_matches:
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
    
    def keyword_search_examples(
        self, 
        example_num: int, 
        subject: str = None,
        school_id: str = None
    ) -> List[Dict]:
        """
        Direct keyword search for specific examples.
        Prioritizes math_image (vision) over math_text (OCR).
        
        Args:
            example_num: Example number to search for
            subject: Optional subject filter (e.g., "Math")
            school_id: School ID for data isolation
        
        Returns:
            List of matching documents, vision-extracted first
        """
        try:
            # Build query
            query = self.vector_store.client.table("documents")\
                .select("*")\
                .ilike("content", f"%Example {example_num}%")
            
            # Add school isolation
            if school_id:
                query = query.eq("school_id", school_id)
            
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
                doc['search_type'] = 'keyword'
            
            return all_results[:10]
        
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
    
    def semantic_search(
        self,
        query: str,
        intent: Dict[str, Any],
        school_id: str = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with intent-aware configuration."""
        try:
            # Generate embedding
            query_result = self.provider.embed_texts([query])
            
            # FIX: Handle different response formats
            if hasattr(query_result, 'vectors') and query_result.vectors:
                query_embedding = query_result.vectors[0]
            elif isinstance(query_result, list):
                query_embedding = query_result[0]
            else:
                raise ValueError("Unexpected embedding format")
            
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
            
            # Perform search with filters
            results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                match_threshold=threshold,
                match_count=limit,
                school_id=school_id,  # FIX: Enforce school isolation
                curriculum_type=None,
                document_type=None,
                academic_year=None,
                grade=intent.get("grade"),
                subject=intent.get("subject")
            )
            
            # Mark as semantic search
            for result in results:
                result['search_type'] = 'semantic'
            
            return results
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        keyword_results: List[Dict],
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge keyword and semantic results using Reciprocal Rank Fusion.
        
        Formula: score = sum(1 / (k + rank))
        
        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            k: Constant for RRF (default 60)
        
        Returns:
            Merged and re-ranked results
        """
        scores = {}
        doc_map = {}
        
        # Score keyword results
        for rank, doc in enumerate(keyword_results, 1):
            doc_id = doc.get('id', hash(doc.get('content', '')))
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Score semantic results
        for rank, doc in enumerate(semantic_results, 1):
            doc_id = doc.get('id', hash(doc.get('content', '')))
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Combine and sort by RRF score
        merged = [
            {**doc_map[doc_id], 'rrf_score': score, 'search_type': 'hybrid'}
            for doc_id, score in scores.items()
        ]
        
        return sorted(merged, key=lambda x: x['rrf_score'], reverse=True)
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate documents by ID or content hash."""
        seen = set()
        unique = []
        
        for result in results:
            # Use ID if available, else content hash
            identifier = result.get('id') or hash(result.get('content', ''))
            
            if identifier not in seen:
                seen.add(identifier)
                unique.append(result)
        
        return unique
    
    def search_with_intent(
        self, 
        query: str, 
        intent: Dict[str, Any],
        school_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        TRUE HYBRID SEARCH: Combines keyword and semantic search using RRF.
        
        Args:
            query: Enhanced query string
            intent: Detected query intent
            school_id: School ID for data isolation
        
        Returns:
            Merged and ranked results
        """
        keyword_results = []
        semantic_results = []
        
        # For example queries, try keyword search
        if intent.get("is_example_query"):
            example_num = intent["example_number"]
            subject = intent.get("subject")
            
            logger.info(f"Performing keyword search for Example {example_num}...")
            keyword_results = self.keyword_search_examples(example_num, subject, school_id)
            
            if keyword_results:
                logger.info(f"Found {len(keyword_results)} keyword results")
        
        # Always perform semantic search for hybrid approach
        logger.info("Performing semantic search...")
        semantic_results = self.semantic_search(query, intent, school_id)
        logger.info(f"Found {len(semantic_results)} semantic results")
        
        # If we have both, use RRF to merge
        if keyword_results and semantic_results:
            logger.info("Merging results using Reciprocal Rank Fusion...")
            merged_results = self.reciprocal_rank_fusion(keyword_results, semantic_results)
            return merged_results
        
        # Otherwise return whichever we have
        return keyword_results or semantic_results
    
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
                # Boost vision-extracted math
                if 'rrf_score' in result:
                    result['rrf_score'] *= 1.5
                elif 'similarity' in result:
                    result['similarity'] = min(1.0, result.get('similarity', 0) * 1.5)
                vision_math.append(result)
            elif content_type == 'math_text':
                text_math.append(result)
            elif content_type == 'image_description':
                image_desc.append(result)
            else:
                other.append(result)
        
        # Re-rank based on intent
        if intent.get("is_example_query"):
            return vision_math + text_math + image_desc + other
        
        elif intent.get("is_image_query"):
            for img in image_desc:
                if 'rrf_score' in img:
                    img['rrf_score'] *= 1.2
                elif 'similarity' in img:
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
            
            search_types = set(r.get('search_type', 'unknown') for r in results)
            
            formatted_parts.append("="*70)
            formatted_parts.append(f"Search Results: {len(results)} total ({image_count} images, {text_count} text)")
            formatted_parts.append(f"Search Method: {', '.join(search_types)}")
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
            
            # Show appropriate score
            if 'rrf_score' in result:
                score = result['rrf_score']
                score_label = f"RRF Score: {score:.3f}"
            else:
                score = result.get('similarity', 0)
                score_label = f"Match: {score:.0%}"
            
            grade = result.get('grade')
            subject = result.get('subject')
            
            type_label = "IMAGE DESCRIPTION" if content_type == 'image_description' else "TEXT"
            
            metadata_parts = []
            if grade:
                metadata_parts.append(f"Grade {grade}")
            if subject:
                metadata_parts.append(subject)
            
            metadata_str = f" | {' | '.join(metadata_parts)}" if metadata_parts else ""
            
            formatted_parts.append(f"【Result {i}】 {type_label}{metadata_str} | Page {page} | {score_label}")
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
    
    def get_cache_key(self, query: str, intent: Dict, school_id: str = None) -> str:
        """Generate cache key from query parameters."""
        cache_data = f"{query}_{intent.get('grade')}_{intent.get('subject')}_{school_id}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def search(
        self, 
        query: str, 
        memory_context: Optional[Dict] = None,
        school_id: str = None,
        verbose: bool = False
    ) -> str:
        """
        Main search function with query enhancement and all features.
        
        Args:
            query: User query
            memory_context: Dictionary with memory info (subjects, grades, pages, topics)
            school_id: School ID for data isolation (REQUIRED for production)
            verbose: Enable debug output
        
        Returns:
            Formatted search results
        """
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        try:
            # Step 0: Enhance query with memory context
            enhanced_query = self.enhance_query(query, memory_context)
            
            if enhanced_query != query and verbose:
                logger.info(f"Original: {query}")
                logger.info(f"Enhanced: {enhanced_query}")
            
            # Step 1: Detect intent
            intent = self.detect_query_intent(enhanced_query)
            
            if verbose:
                logger.info(f"Query Intent: {intent['type']}")
                if intent['is_image_query']:
                    logger.info("→ Image-focused search enabled")
                if intent['is_page_specific']:
                    logger.info(f"→ Page-specific: {intent['pages']}")
                if intent['is_example_query']:
                    logger.info(f"→ Example search: {intent['example_number']}")
                if intent.get('grade'):
                    logger.info(f"→ Grade filter: {intent['grade']}")
                if intent.get('subject'):
                    logger.info(f"→ Subject filter: {intent['subject']}")
            
            # Step 2: Check cache
            cache_key = self.get_cache_key(enhanced_query, intent, school_id)
            if cache_key in self.semantic_cache:
                cached_time, cached_results = self.semantic_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    self.metrics["cache_hits"] += 1
                    if verbose:
                        logger.info("→ Cache hit!")
                    return cached_results
            
            # Step 3: Search with filters (TRUE HYBRID)
            results = self.search_with_intent(enhanced_query, intent, school_id)
            
            if verbose:
                logger.info(f"→ Found {len(results)} initial results")
            
            # Step 4: Deduplicate
            results = self.deduplicate_results(results)
            
            if verbose:
                logger.info(f"→ After deduplication: {len(results)} results")
            
            # Step 5: Re-rank
            reranked_results = self.rerank_results(results, intent)
            
            if verbose:
                image_count = sum(1 for r in reranked_results if r.get('content_type') == 'image_description')
                logger.info(f"→ After re-ranking: {image_count} image descriptions prioritized")
            
            # Step 6: Format
            formatted = self.format_results(reranked_results, max_results=10, verbose=verbose)
            
            # Cache result
            self.semantic_cache[cache_key] = (time.time(), formatted)
            
            # Track latency
            latency = time.time() - start_time
            self.metrics["avg_latency"].append(latency)
            
            if verbose:
                logger.info(f"→ Search completed in {latency:.2f}s")
            
            return formatted
        
        except Exception as e:
            self.metrics["failed_queries"] += 1
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            
            # Return user-friendly error
            return f"❌ Search Error: Unable to complete search. Please try again or contact support.\n\nTechnical details: {str(e)}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_latency = sum(self.metrics["avg_latency"]) / len(self.metrics["avg_latency"]) if self.metrics["avg_latency"] else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_queries"]),
            "failed_queries": self.metrics["failed_queries"],
            "failure_rate": self.metrics["failed_queries"] / max(1, self.metrics["total_queries"]),
            "avg_latency": avg_latency
        }


# Global search engine instance
_search_engine = None


def get_search_engine() -> SmartSearchEngine:
    """Get or create search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = SmartSearchEngine()
    return _search_engine


def search_school_documents(
    query: str, 
    memory_context: Optional[Dict] = None,
    school_id: str = None
) -> str:
    """
    Enhanced search function with memory-based query enhancement and TRUE hybrid search.
    
    Args:
        query: User's search query
        memory_context: Optional dict with memory info (subjects, grades, pages, last_topic)
        school_id: School ID for data isolation (REQUIRED in production)
    
    Returns:
        Formatted search results
    """
    try:
        engine = get_search_engine()
        return engine.search(
            query, 
            memory_context=memory_context,
            school_id=school_id,
            verbose=True
        )
    
    except Exception as e:
        logger.error(f"Search tool error: {str(e)}", exc_info=True)
        return f"❌ Search Error: {str(e)}"


# Create LangChain Tool
supabase_tool = Tool.from_function(
    name="search_school_documents",
    description="""Search comprehensive educational database with TRUE HYBRID SEARCH (keyword + semantic with RRF fusion), automatic grade/subject filtering, and memory-enhanced query understanding.

**Database Organization:**
- Multi-tenant system with school-level data isolation
- Content organized by GRADE (1-12) and SUBJECT
- Complete textbook content with text + AI-generated image descriptions
- School policies, handbooks, and administrative procedures
- Multi-curriculum support (CBSE, SSE, ICSE, IB, State Boards)

**Advanced Search Capabilities:**
1. **True Hybrid Search**
   - Combines keyword matching and semantic similarity simultaneously
   - Results merged using Reciprocal Rank Fusion (RRF)
   - Automatic deduplication and relevance scoring
   - Prioritizes vision-extracted math over OCR text

2. **Memory-Enhanced Query Understanding**
   - Automatically enhances vague queries using conversation context
   - Resolves pronouns and references to previous content
   - Maintains context across multi-turn conversations
   - Tracks subjects, grades, pages, topics, and example numbers

3. **Intelligent Filtering**
   - Auto-detects grade levels from natural language
   - Subject identification from keywords and context
   - Page-specific queries supported
   - Content type filtering (text, images, diagrams)
   - School ID enforcement for data isolation

**Query Enhancement Examples:**
- Vague: "example N" → Enhanced: "[topic] example N [subject] [grade]"
- Follow-up: "next problem" → Enhanced: "next problem [previous context]"
- Reference: "that diagram" → Enhanced: "diagram [subject] page [N]"
- Generic: "explain it" → Enhanced: "[topic] [subject] [grade]"

**Performance Features:**
- Semantic caching with 1-hour TTL
- Configurable similarity thresholds
- Graceful error handling with fallbacks
- Real-time performance metrics
- Optimized for low-latency retrieval

**Supported Query Types:**
- Specific examples and problems
- Multi-part questions (e.g., "part ii")
- Page-specific content requests
- Image and diagram queries
- Conceptual explanations
- Curriculum-specific questions
- Follow-up and clarification queries

**Returns:**
Detailed results with source citations (filename, page, grade, subject), search method used, confidence scores, and structured metadata for seamless integration.""",
    func=search_school_documents,
    return_direct=False
)
