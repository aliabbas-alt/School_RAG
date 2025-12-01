# enhanced_conversation_memory.py
"""
Universal Conversation Memory for School RAG
Works for ALL subjects: Math, Science, History, English, etc.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import json
import os
import re
import logging

logger = logging.getLogger(__name__)


class EnhancedConversationMemory:
    """
    Universal memory system for all educational subjects.
    
    Key features:
    - Subject-agnostic concept extraction
    - Reference resolution (pronouns, "the two", "it", etc.)
    - Works for any curriculum (CBSE, ICSE, State boards, etc.)
    - Handles any subject (Math, Science, History, Languages, etc.)
    """
    
    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        
        # RAW HISTORY
        self.history: List[Dict[str, Any]] = []
        
        # UNIVERSAL CONCEPT TRACKING (subject-independent)
        self.current_concepts: List[str] = []  # Stack of recent concepts
        self.last_topic: str = ""
        self.last_two_topics: List[str] = []  # For comparison queries
        
        # ENTITY TRACKING (universal)
        self.subjects_discussed: Set[str] = set()
        self.chapters_mentioned: Set[str] = set()
        self.pages_mentioned: Set[int] = set()
        self.examples_mentioned: Set[int] = set()
        
        # METADATA
        self.metadata: Dict[str, Any] = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
        
        logger.info("Universal conversation memory initialized")
    
    def _extract_noun_phrases(self, text: str) -> Set[str]:
        """
        Extract noun phrases as concepts (subject-agnostic).
        
        This works for ANY subject:
        - Math: "derivatives", "integrals", "quadratic equations"
        - Science: "photosynthesis", "mitochondria", "chemical reactions"
        - History: "world war", "independence movement", "mughal empire"
        - English: "metaphors", "sonnets", "passive voice"
        """
        concepts = set()
        
        # Method 1: Capitalized phrases (proper nouns, important terms)
        # Works for: "World War II", "Pythagorean Theorem", "French Revolution"
        capitalized_phrases = re.findall(
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            text
        )
        concepts.update(phrase.lower() for phrase in capitalized_phrases if len(phrase) > 3)
        
        # Method 2: Quoted terms (explicit concepts)
        # Works for: "photosynthesis", 'cellular respiration'
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        concepts.update(term.lower() for term in quoted if len(term) > 3)
        
        # Method 3: Technical terms (words with specific patterns)
        # Works for: compound words, hyphenated terms
        technical = re.findall(r'\b[a-z]+(?:-[a-z]+)+\b', text.lower())
        concepts.update(technical)
        
        # Method 4: Important keywords (based on context clues)
        # Look for phrases like "what is X", "explain X", "define X"
        definition_patterns = [
            r'(?:what is|explain|define|describe)\s+(?:a |an |the )?([a-z\s]+?)(?:\?|\.|\,)',
            r'(?:tell me about|concept of|theory of)\s+(?:a |an |the )?([a-z\s]+?)(?:\?|\.|\,)',
        ]
        for pattern in definition_patterns:
            matches = re.findall(pattern, text.lower())
            concepts.update(m.strip() for m in matches if len(m.strip()) > 3)
        
        # Method 5: Detect subject-specific terms by word frequency
        # Terms that appear multiple times are likely important concepts
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        word_counts = Counter(words)
        frequent_terms = {word for word, count in word_counts.items() if count >= 2}
        concepts.update(frequent_terms)
        
        # Filter out common stopwords
        stopwords = {
            'this', 'that', 'these', 'those', 'what', 'which', 'when', 'where',
            'who', 'whom', 'whose', 'how', 'why', 'about', 'from', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'between',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'could', 'should', 'would', 'might', 'must', 'shall', 'will',
            'does', 'have', 'been', 'being', 'were', 'was', 'are', 'is',
            'the', 'and', 'or', 'but', 'for', 'nor', 'yet', 'so'
        }
        
        concepts = {c for c in concepts if c not in stopwords and len(c) > 2}
        
        return concepts
    
    def _detect_reference_query(self, query: str) -> Tuple[bool, str]:
        """
        Detect if query references previous concepts (universal).
        Works for all subjects.
        """
        query_lower = query.lower().strip()
        
        reference_patterns = {
            "comparison": [
                r'\bdifference between (?:the )?two\b',
                r'\bcompare (?:the )?two\b',
                r'\bcompare them\b',
                r'\bthem\b',
                r'\bthese two\b',
                r'\bboth of them\b',
                r'\bone and (?:the )?other\b',
                r'\bversus\b',
                r'\bvs\b',
                r'\bhow (?:are they|do they) different\b',
                r'\bsimilarities and differences\b'
            ],
            "single_reference": [
                r'^it\b',
                r'\bit\b',
                r'^this\b',
                r'\bthis\b',
                r'^that\b',
                r'\bthat\b',
                r'\bthe same\b',
                r'\bthe concept\b',
                r'\bthe topic\b',
                r'\bthe above\b',
                r'\bthe previous\b',
                r'\bthe mentioned\b'
            ],
            "follow_up": [
                r'^(?:and|also|what about|how about)',
                r'^(?:can you|could you|will you|would you)',
                r'^(?:explain|tell me|give me) (?:more|further|another)',
                r'\bmore details?\b',
                r'\belaborate\b',
                r'\bexamples?\b',
                r'^why\b',
                r'^how\b'
            ]
        }
        
        for ref_type, patterns in reference_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return True, ref_type
        
        return False, "none"
    
    def _resolve_reference(self, query: str, ref_type: str) -> str:
        """
        Resolve pronouns/references to explicit concepts (universal).
        
        Examples:
        - "What's the difference between the two?" 
          → "What's the difference between photosynthesis and respiration?"
        - "Explain it in detail" 
          → "Explain mitochondria in detail"
        - "And World War II?" 
          → "Explain World War II"
        """
        if ref_type == "comparison" and len(self.last_two_topics) >= 2:
            concept1, concept2 = self.last_two_topics[-2], self.last_two_topics[-1]
            
            resolved = query
            
            # Replace "the two", "them", "these two", etc.
            replacements = [
                (r'\b(?:the )?two\b', f"{concept1} and {concept2}"),
                (r'\bthem\b', f"{concept1} and {concept2}"),
                (r'\bthese two\b', f"{concept1} and {concept2}"),
                (r'\bboth of them\b', f"{concept1} and {concept2}"),
                (r'\bone and (?:the )?other\b', f"{concept1} and {concept2}"),
            ]
            
            for pattern, replacement in replacements:
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)
            
            logger.info(f"🔗 Resolved comparison: '{query}' → '{resolved}'")
            return resolved
        
        elif ref_type == "single_reference" and self.last_topic:
            resolved = query
            
            # Replace "it", "this", "that"
            replacements = [
                (r'^it\b', self.last_topic),
                (r'\bit\b', self.last_topic),
                (r'^this\b', self.last_topic),
                (r'\bthis\b', self.last_topic),
                (r'^that\b', self.last_topic),
                (r'\bthat\b', self.last_topic),
                (r'\bthe concept\b', self.last_topic),
                (r'\bthe topic\b', self.last_topic),
            ]
            
            for pattern, replacement in replacements:
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE, count=1)
            
            logger.info(f"🔗 Resolved reference: '{query}' → '{resolved}'")
            return resolved
        
        elif ref_type == "follow_up" and self.last_topic:
            # For follow-ups like "And photosynthesis?", extract the new topic
            # but keep context
            pass
        
        return query
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract universal metadata (works for all subjects).
        IMPROVED: Separate query vs response extraction.
        
        Args:
            text: Combined user query + assistant response
        
        Returns:
            Dictionary with extracted metadata
        """
        text_lower = text.lower()
        
        metadata = {
            "pages": set(),
            "chapters": set(),
            "examples": set(),
            "grades": set(),
            "concepts": set()
        }
        
        # Extract pages (universal)
        page_patterns = [
            r'\bpage\s+(\d+)\b',
            r'\bp\.?\s*(\d+)\b',
            r'\bpg\.?\s*(\d+)\b'
        ]
        for pattern in page_patterns:
            for match in re.findall(pattern, text_lower):
                metadata["pages"].add(int(match))
        
        # Extract chapters (universal)
        chapter_patterns = [
            r'\bchapter\s+(\d+)\b',
            r'\bch\.?\s*(\d+)\b',
            r'\bunit\s+(\d+)\b'
        ]
        for pattern in chapter_patterns:
            matches = re.findall(pattern, text_lower)
            metadata["chapters"].update(f"chapter_{m}" for m in matches)
        
        # Extract chapter names (e.g., "Chapter: Photosynthesis")
        chapter_name_pattern = r'chapter[:\s]+([a-z\s]+?)(?:\n|\.|\?)'
        chapter_names = re.findall(chapter_name_pattern, text_lower)
        metadata["chapters"].update(name.strip() for name in chapter_names)
        
        # Extract examples (universal)
        example_patterns = [
            r'\bexample\s+(\d+)\b',
            r'\bex\.?\s*(\d+)\b',
            r'\bquestion\s+(\d+)\b',
            r'\bq\.?\s*(\d+)\b'
        ]
        for pattern in example_patterns:
            for match in re.findall(pattern, text_lower):
                metadata["examples"].add(int(match))
        
        # Extract grades/classes (universal)
        grade_patterns = [
            r'\bgrade\s+(\d+)\b',
            r'\bclass\s+(\d+)\b',
            r'\bstd\.?\s*(\d+)\b'
        ]
        for pattern in grade_patterns:
            for match in re.findall(pattern, text_lower):
                grade_num = int(match)
                if 1 <= grade_num <= 12:
                    metadata["grades"].add(grade_num)
        
        # Extract concepts (subject-agnostic)
        question_part = text[:200] if len(text) > 200 else text
        metadata["concepts"] = self._extract_noun_phrases(question_part)
        
        return metadata
    
    def add_interaction(
        self,
        user_query: str,
        assistant_response: str,
        retrieved_docs: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add interaction with universal context tracking.
        
        Args:
            user_query: User's question (may contain pronouns/references)
            assistant_response: Assistant's answer
            retrieved_docs: Documents retrieved (optional)
        """
        # 1. Detect references
        is_reference, ref_type = self._detect_reference_query(user_query)
        
        # 2. Resolve references
        resolved_query = user_query
        if is_reference:
            resolved_query = self._resolve_reference(user_query, ref_type)
            logger.info(f"Reference type: {ref_type}")
        
        # 3. Extract metadata (universal)
        combined_text = resolved_query + " " + assistant_response
        metadata = self._extract_metadata(combined_text)
        
        # 4. Update concept tracking
        new_concepts = list(metadata["concepts"])[:5]  # Top 5 concepts
        
        if new_concepts:
            # Update concept stack
            self.current_concepts.extend(new_concepts)
            self.current_concepts = self.current_concepts[-10:]  # Keep last 10
            
            # Update last two topics (for comparisons)
            self.last_two_topics.extend(new_concepts)
            self.last_two_topics = self.last_two_topics[-2:]
            
            # Update last topic
            self.last_topic = new_concepts[0]
        
        # 5. Store interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "resolved_user": resolved_query,
            "assistant": assistant_response,
            "docs_retrieved": len(retrieved_docs) if retrieved_docs else 0,
            "is_reference": is_reference,
            "reference_type": ref_type if is_reference else None,
            "metadata": {
                "pages": sorted(list(metadata["pages"])),
                "chapters": sorted(list(metadata["chapters"])),
                "examples": sorted(list(metadata["examples"])),
                "grades": sorted(list(metadata["grades"])),
                "concepts": list(metadata["concepts"])[:10]  # Top 10
            }
        }
        
        self.history.append(interaction)
        self.metadata["total_queries"] += 1
        
        # Keep only last N
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Update tracked entities
        self.subjects_discussed.update(metadata["concepts"])
        self.chapters_mentioned.update(metadata["chapters"])
        self.pages_mentioned.update(metadata["pages"])
        self.examples_mentioned.update(metadata["examples"])
        
        # Log
        logger.info(f"💬 Interaction added:")
        logger.info(f"   Concepts: {new_concepts}")
        logger.info(f"   Last two: {self.last_two_topics}")
        if metadata["pages"]:
            logger.info(f"   Pages: {sorted(metadata['pages'])}")
        if metadata["chapters"]:
            logger.info(f"   Chapters: {metadata['chapters']}")
    
    def get_context_for_query(self, current_query: str, include_n: int = 3) -> str:
        """
        Get rich context for current query (universal).
        
        THIS IS THE METHOD YOUR AGENT SHOULD USE!
        
        Args:
            current_query: Current user query
            include_n: Number of recent interactions to include
        
        Returns:
            Rich context string with resolved concepts
        """
        if not self.history:
            return "No previous conversation."
        
        # Detect references
        is_reference, ref_type = self._detect_reference_query(current_query)
        
        context_parts = []
        
        # 1. Add explicit concept context for references
        if is_reference:
            context_parts.append("=== IMPORTANT CONTEXT ===")
            
            if ref_type == "comparison" and len(self.last_two_topics) >= 2:
                context_parts.append(
                    f"The user is asking to compare: "
                    f"'{self.last_two_topics[-2]}' and '{self.last_two_topics[-1]}'"
                )
                context_parts.append("(from previous questions)\n")
            
            elif ref_type == "single_reference" and self.last_topic:
                context_parts.append(
                    f"The user is referring to: '{self.last_topic}'"
                )
                context_parts.append("(from the previous question)\n")
        
        # 2. Add recent concept stack
        if self.current_concepts:
            context_parts.append(
                f"Recent topics: {', '.join(self.current_concepts[-5:])}\n"
            )
        
        # 3. Add conversation history
        context_parts.append("=== Recent Conversation ===\n")
        
        recent = self.history[-include_n:]
        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {interaction['user']}")
            
            # Show resolved query if different
            if interaction.get('is_reference') and interaction['resolved_user'] != interaction['user']:
                context_parts.append(f"[Resolved: {interaction['resolved_user']}]")
            
            # Show key concepts
            concepts = interaction['metadata'].get('concepts', [])[:3]
            if concepts:
                context_parts.append(f"Key concepts: {', '.join(concepts)}")
            
            # Show assistant response (truncated)
            context_parts.append(f"Assistant: {interaction['assistant'][:200]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_context_for_enhancement(self) -> Dict[str, Any]:
        """Get structured context for query enhancement (universal)."""
        return {
            "recent_concepts": self.current_concepts[-5:],
            "last_topic": self.last_topic,
            "last_two_topics": self.last_two_topics,
            "pages_mentioned": sorted(list(self.pages_mentioned)),
            "chapters_mentioned": sorted(list(self.chapters_mentioned)),
            "examples_mentioned": sorted(list(self.examples_mentioned))
        }
    
    def get_summary(self) -> str:
        """Get conversation summary (universal)."""
        parts = []
        
        if self.current_concepts:
            parts.append(f"💡 Recent topics: {', '.join(self.current_concepts[-5:])}")
        
        if self.last_two_topics:
            parts.append(f"🔄 Last two: {' & '.join(self.last_two_topics)}")
        
        if self.chapters_mentioned:
            chapters = list(self.chapters_mentioned)[:5]
            parts.append(f"📚 Chapters: {', '.join(chapters)}")
        
        if self.pages_mentioned:
            pages = sorted(list(self.pages_mentioned))
            if len(pages) <= 5:
                parts.append(f"📄 Pages: {', '.join(map(str, pages))}")
            else:
                parts.append(f"📄 Pages: {min(pages)}-{max(pages)}")
        
        parts.append(f"💬 Interactions: {len(self.history)}")
        
        return "\n".join(parts) if parts else "No conversation history"
    
    def clear(self) -> None:
        """Clear all memory."""
        self.history.clear()
        self.current_concepts.clear()
        self.subjects_discussed.clear()
        self.chapters_mentioned.clear()
        self.pages_mentioned.clear()
        self.examples_mentioned.clear()
        self.last_topic = ""
        self.last_two_topics.clear()
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
