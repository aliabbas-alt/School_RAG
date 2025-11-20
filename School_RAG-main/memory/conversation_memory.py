# enhanced_conversation_memory.py
"""
Advanced Hybrid Memory System for School RAG - Production Ready
Combines entity extraction, attention mechanism, and temporal weighting.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import json
import os
import re
import logging
import math

logger = logging.getLogger(__name__)


class EnhancedConversationMemory:
    """
    Advanced memory system combining:
    1. Entity extraction (subjects, grades, pages, topics) for query enhancement
    2. Attention-based retrieval (DMN-inspired)
    3. Temporal decay weighting
    4. Episode segmentation (study sessions)
    5. Query type classification for analytics
    6. Session persistence for recovery
    """
    
    def __init__(self, max_history: int = 15, episode_gap_minutes: int = 30):
        self.max_history = max_history
        self.episode_gap_minutes = episode_gap_minutes
        
        # RAW HISTORY
        self.history: List[Dict[str, Any]] = []
        
        # EXTRACTED ENTITIES
        self.subjects_discussed: Set[str] = set()
        self.grades_discussed: Set[int] = set()
        self.pages_mentioned: Set[int] = set()
        self.last_topic: str = ""
        self.last_example: Optional[int] = None
        self.last_page: Optional[int] = None
        
        # ANALYTICS
        self.question_count = 0
        self.clarification_count = 0
        self.query_types: List[str] = []
        
        # EPISODES (Study Sessions)
        self.episodes: List[Dict] = []
        self.current_episode_start: Optional[datetime] = None
        
        # CONCEPT GRAPH (Multi-hop reasoning)
        self.concept_graph: Dict[str, Set[str]] = defaultdict(set)
        self.concept_co_occurrence: Dict[tuple, int] = defaultdict(int)
        
        # METADATA
        self.metadata: Dict[str, Any] = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
        
        # Subject keywords
        self.subject_keywords = {
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
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract subjects, grades, pages, topics from text."""
        text_lower = text.lower()
        
        entities = {
            "subjects": set(),
            "grades": set(),
            "pages": set(),
            "topics": [],
            "example_num": None
        }
        
        # Extract subjects
        for subject, keywords in self.subject_keywords.items():
            if any(kw in text_lower for kw in keywords):
                entities["subjects"].add(subject)
        
        # Extract grades
        grade_pattern = r'\bgrade\s+(\d+)\b|\bclass\s+(\d+)\b'
        for match in re.findall(grade_pattern, text_lower):
            num_str = match[0] or match[1]
            grade_num = int(num_str)
            if 1 <= grade_num <= 12:
                entities["grades"].add(grade_num)
        
        # Extract pages
        page_pattern = r'\bpage\s+(\d+)\b|\bp\.?\s*(\d+)\b'
        for match in re.findall(page_pattern, text_lower):
            page_num = int(match[0] or match[1])
            entities["pages"].add(page_num)
        
        # Extract topics
        topic_keywords = [
            "integral", "derivative", "limit", "series", "matrix", "vector",
            "equation", "function", "theorem", "proof", "formula", "calculus"
        ]
        entities["topics"] = [t for t in topic_keywords if t in text_lower]
        
        # Extract example number
        example_match = re.search(r'example\s+(\d+)', text_lower)
        if example_match:
            entities["example_num"] = int(example_match.group(1))
        
        return entities
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for analytics."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "explanation"
        elif any(word in query_lower for word in ["how", "steps", "process"]):
            return "procedural"
        elif any(word in query_lower for word in ["image", "diagram", "picture", "figure"]):
            return "visual"
        elif re.search(r'\bexample\s+\d+\b', query_lower):
            return "example"
        elif query_lower.startswith(("can", "could", "would")):
            self.clarification_count += 1
            return "clarification"
        else:
            return "general"
    
    def _build_concept_graph(self, entities: Dict[str, Any]) -> None:
        """Build graph of co-occurring concepts for multi-hop reasoning."""
        concepts = []
        
        # Collect all concepts
        concepts.extend(entities['subjects'])
        concepts.extend(f"grade_{g}" for g in entities['grades'])
        concepts.extend(f"page_{p}" for p in entities['pages'])
        concepts.extend(entities['topics'])
        
        # Update co-occurrence
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                pair = tuple(sorted([concept_a, concept_b]))
                self.concept_co_occurrence[pair] += 1
                self.concept_graph[concept_a].add(concept_b)
                self.concept_graph[concept_b].add(concept_a)
    
    def _check_episode_boundary(self) -> bool:
        """Check if we should start a new episode (study session)."""
        if not self.history:
            return True
        
        last_interaction = self.history[-1]
        last_time = datetime.fromisoformat(last_interaction['timestamp'])
        gap = (datetime.now() - last_time).total_seconds() / 60
        
        return gap > self.episode_gap_minutes
    
    def add_interaction(
        self,
        user_query: str,
        assistant_response: str,
        retrieved_docs: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add interaction with advanced processing.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            retrieved_docs: Documents retrieved (optional)
        """
        # Extract entities first
        combined_text = user_query + " " + assistant_response
        entities = self._extract_entities(combined_text)
        
        # Classify query
        query_type = self._classify_query_type(user_query)
        
        # Check episode boundary
        if self._check_episode_boundary():
            if self.current_episode_start and self.history:
                # Save completed episode
                self.episodes.append({
                    "start": self.current_episode_start.isoformat(),
                    "end": datetime.now().isoformat(),
                    "interaction_count": len([h for h in self.history 
                                              if datetime.fromisoformat(h['timestamp']) >= self.current_episode_start]),
                    "subjects": list(self.subjects_discussed),
                    "grades": sorted(list(self.grades_discussed))
                })
            self.current_episode_start = datetime.now()
        
        # Store raw interaction with entities
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "assistant": assistant_response,
            "docs_retrieved": len(retrieved_docs) if retrieved_docs else 0,
            "query_type": query_type,
            "entities": {
                "pages": sorted(list(entities["pages"])),
                "subjects": list(entities["subjects"]),
                "grades": sorted(list(entities["grades"])),
                "topics": entities["topics"]
            }
        }
        
        self.history.append(interaction)
        self.query_types.append(query_type)
        self.metadata["total_queries"] += 1
        self.question_count += 1
        
        # Keep only last N
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Update tracked entities
        self.subjects_discussed.update(entities["subjects"])
        self.grades_discussed.update(entities["grades"])
        self.pages_mentioned.update(entities["pages"])
        
        if entities["topics"]:
            self.last_topic = entities["topics"][0]
        
        if entities["example_num"]:
            self.last_example = entities["example_num"]
        
        if entities["pages"]:
            self.last_page = max(entities["pages"])
        
        # Build concept graph
        self._build_concept_graph(entities)
        
        # Log extracted entities
        if entities["subjects"]:
            logger.info(f"Detected subjects: {entities['subjects']}")
        if entities["grades"]:
            logger.info(f"Detected grades: {entities['grades']}")
        if entities["pages"]:
            logger.info(f"Detected pages: {entities['pages']}")
    
    def get_context_for_enhancement(self) -> Dict[str, Any]:
        """
        Get structured context for query enhancement.
        Used by supabase_tool.enhance_query()
        """
        return {
            "subjects_discussed": self.subjects_discussed,
            "grades_discussed": self.grades_discussed,
            "pages_mentioned": self.pages_mentioned,
            "last_topic": self.last_topic,
            "last_example": self.last_example,
            "last_page": self.last_page
        }
    
    def get_conversation_context(self, last_n: int = 3) -> str:
        """
        Get conversation history for LLM context window.
        Used in agent prompts.
        """
        if not self.history:
            return "No previous conversation."
        
        recent = self.history[-last_n:]
        context_parts = ["=== Recent Conversation History ===\n"]
        
        for i, interaction in enumerate(recent, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['assistant'][:200]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_smart_context(self, current_query: str, last_n: int = 3) -> str:
        """
        Get context relevant to current query using entity overlap.
        Falls back to recent history if no overlap found.
        
        Args:
            current_query: Current user query
            last_n: Number of interactions to return
        
        Returns:
            Formatted context string
        """
        if not self.history:
            return "No previous conversation."
        
        # Extract entities from current query
        current_entities = self._extract_entities(current_query)
        
        # Find interactions with entity overlap
        relevant = []
        for interaction in reversed(self.history[-10:]):
            if 'entities' in interaction:
                past_entities = interaction['entities']
                
                # Check overlaps
                page_overlap = bool(current_entities['pages'] & set(past_entities.get('pages', [])))
                subject_overlap = bool(current_entities['subjects'] & set(past_entities.get('subjects', [])))
                grade_overlap = bool(current_entities['grades'] & set(past_entities.get('grades', [])))
                
                if page_overlap or subject_overlap or grade_overlap:
                    relevant.append(interaction)
                    if len(relevant) >= last_n:
                        break
        
        # If no relevant history, use recent
        if not relevant:
            return self.get_conversation_context(last_n)
        
        # Format relevant context
        context_parts = ["=== Relevant Context (Entity-Based) ===\n"]
        
        for i, interaction in enumerate(relevant, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['assistant'][:200]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_attention_weighted_context(self, current_query: str, top_k: int = 3) -> str:
        """
        Use attention mechanism to find most relevant past interactions.
        Inspired by Dynamic Memory Networks.
        
        Args:
            current_query: Current user query
            top_k: Number of most relevant interactions to return
        
        Returns:
            Attention-weighted context string
        """
        if not self.history:
            return "No previous conversation."
        
        # Extract query entities
        query_entities = self._extract_entities(current_query)
        query_lower = current_query.lower()
        
        # Score each interaction
        scored_interactions: List[Tuple[Dict, float]] = []
        
        for interaction in self.history[-20:]:  # Look at last 20
            score = 0.0
            
            # 1. Entity overlap score (40% weight)
            if 'entities' in interaction:
                past_entities = interaction['entities']
                
                # Page overlap (high weight - very specific)
                page_overlap = len(query_entities['pages'] & set(past_entities.get('pages', [])))
                score += page_overlap * 0.15
                
                # Subject overlap (medium weight)
                subject_overlap = len(query_entities['subjects'] & set(past_entities.get('subjects', [])))
                score += subject_overlap * 0.10
                
                # Grade overlap (medium weight)
                grade_overlap = len(query_entities['grades'] & set(past_entities.get('grades', [])))
                score += grade_overlap * 0.10
                
                # Topic overlap (low weight)
                topic_overlap = len(set(query_entities['topics']) & set(past_entities.get('topics', [])))
                score += topic_overlap * 0.05
            
            # 2. Lexical overlap score (30% weight)
            user_words = set(interaction['user'].lower().split())
            query_words = set(query_lower.split())
            lexical_overlap = len(user_words & query_words) / max(len(query_words), 1)
            score += lexical_overlap * 0.30
            
            # 3. Recency score (20% weight)
            position = self.history.index(interaction)
            recency = position / len(self.history)
            score += recency * 0.20
            
            # 4. Query type similarity (10% weight)
            current_type = self._classify_query_type(current_query)
            past_type = interaction.get('query_type', 'general')
            if current_type == past_type:
                score += 0.10
            
            scored_interactions.append((interaction, score))
        
        # Sort by score and take top_k
        scored_interactions.sort(key=lambda x: x[1], reverse=True)
        top_interactions = scored_interactions[:top_k]
        
        if not top_interactions:
            return self.get_conversation_context(3)
        
        # Format with attention scores
        context_parts = ["=== Attention-Weighted Context ===\n"]
        
        for i, (interaction, score) in enumerate(top_interactions, 1):
            context_parts.append(f"Turn {i} (relevance: {score:.2f}):")
            context_parts.append(f"User: {interaction['user']}")
            context_parts.append(f"Assistant: {interaction['assistant'][:200]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_temporal_weighted_context(self, current_query: str, decay_rate: float = 0.1) -> str:
        """
        Apply temporal decay - recent interactions weighted higher.
        
        Args:
            current_query: Current user query
            decay_rate: How quickly to decay (0 = no decay, 1 = fast decay)
        
        Returns:
            Temporally weighted context
        """
        if not self.history:
            return "No previous conversation."
        
        current_time = datetime.now()
        weighted_interactions: List[Tuple[Dict, float]] = []
        
        for interaction in self.history[-10:]:
            # Calculate time decay
            timestamp = datetime.fromisoformat(interaction['timestamp'])
            time_diff = (current_time - timestamp).total_seconds() / 3600  # hours
            decay_weight = math.exp(-decay_rate * time_diff)
            
            # Entity relevance
            query_entities = self._extract_entities(current_query)
            if 'entities' in interaction:
                past_entities = interaction['entities']
                relevance = (
                    len(query_entities['pages'] & set(past_entities.get('pages', []))) * 0.4 +
                    len(query_entities['subjects'] & set(past_entities.get('subjects', []))) * 0.3 +
                    len(query_entities['grades'] & set(past_entities.get('grades', []))) * 0.3
                )
            else:
                relevance = 0
            
            # Combined score
            final_score = decay_weight * (1 + relevance)
            weighted_interactions.append((interaction, final_score))
        
        # Sort by combined score
        weighted_interactions.sort(key=lambda x: x[1], reverse=True)
        
        # Format top 3
        context_parts = ["=== Temporal Memory (Recent + Relevant) ===\n"]
        for inter, score in weighted_interactions[:3]:
            context_parts.append(f"[Score: {score:.2f}] {inter['user']}")
            context_parts.append(f"→ {inter['assistant'][:150]}...\n")
        
        return "\n".join(context_parts)
    
    def get_related_concepts(self, concept: str, max_depth: int = 2) -> Set[str]:
        """
        Get concepts related through multi-hop connections.
        
        Args:
            concept: Starting concept
            max_depth: Maximum hops in graph
        
        Returns:
            Set of related concepts
        """
        if concept not in self.concept_graph:
            return set()
        
        related = set()
        current_level = {concept}
        visited = {concept}
        
        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = self.concept_graph.get(node, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        related.add(neighbor)
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current_level = next_level
            if not current_level:
                break
        
        return related
    
    def get_episode_summary(self) -> str:
        """Get summary of learning episodes (study sessions)."""
        if not self.episodes:
            return "Current episode in progress."
        
        parts = ["📚 Learning Episodes:\n"]
        for i, ep in enumerate(self.episodes[-5:], 1):
            duration = (datetime.fromisoformat(ep['end']) - 
                       datetime.fromisoformat(ep['start'])).total_seconds() / 60
            parts.append(f"Episode {i}: {duration:.0f} min, {ep['interaction_count']} questions")
            parts.append(f"  Topics: {', '.join(ep['subjects'][:3])}")
        
        return "\n".join(parts)
    
    def get_summary(self) -> str:
        """Get detailed summary of conversation state."""
        parts = []
        
        if self.subjects_discussed:
            parts.append(f"📚 Subjects: {', '.join(self.subjects_discussed)}")
        
        if self.grades_discussed:
            parts.append(f"🎓 Grades: {', '.join(map(str, sorted(self.grades_discussed)))}")
        
        if self.pages_mentioned:
            pages = sorted(list(self.pages_mentioned))
            if len(pages) <= 5:
                parts.append(f"📄 Pages: {', '.join(map(str, pages))}")
            else:
                parts.append(f"📄 Pages: {len(pages)} pages ({min(pages)}-{max(pages)})")
        
        if self.last_topic:
            parts.append(f"💡 Last topic: {self.last_topic}")
        
        if self.last_example:
            parts.append(f"📝 Last example: {self.last_example}")
        
        if self.last_page:
            parts.append(f"📖 Last page: {self.last_page}")
        
        parts.append(f"💬 Interactions: {len(self.history)}")
        
        # Query type analytics
        if self.query_types:
            type_counts = Counter(self.query_types)
            most_common = type_counts.most_common(1)[0]
            parts.append(f"🔍 Most common: {most_common[0]} ({most_common[1]}×)")
        
        # Episodes
        if self.episodes:
            parts.append(f"📅 Study sessions: {len(self.episodes)}")
        
        return "\n".join(parts) if parts else "No conversation history"
    
    def clear(self) -> None:
        """Clear all memory."""
        self.history.clear()
        self.subjects_discussed.clear()
        self.grades_discussed.clear()
        self.pages_mentioned.clear()
        self.last_topic = ""
        self.last_example = None
        self.last_page = None
        self.question_count = 0
        self.clarification_count = 0
        self.query_types.clear()
        self.episodes.clear()
        self.current_episode_start = None
        self.concept_graph.clear()
        self.concept_co_occurrence.clear()
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save everything to JSON."""
        data = {
            "metadata": {
                **self.metadata,
                "question_count": self.question_count,
                "clarification_count": self.clarification_count
            },
            "history": self.history,
            "extracted_entities": {
                "subjects_discussed": list(self.subjects_discussed),
                "grades_discussed": sorted(list(self.grades_discussed)),
                "pages_mentioned": sorted(list(self.pages_mentioned)),
                "last_topic": self.last_topic,
                "last_example": self.last_example,
                "last_page": self.last_page
            },
            "analytics": {
                "query_types": self.query_types,
                "type_distribution": dict(Counter(self.query_types))
            },
            "episodes": self.episodes,
            "concept_graph": {k: list(v) for k, v in self.concept_graph.items()}
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """Load everything from JSON."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.history = data.get("history", [])
        self.question_count = self.metadata.get("question_count", 0)
        self.clarification_count = self.metadata.get("clarification_count", 0)
        
        entities = data.get("extracted_entities", {})
        self.subjects_discussed = set(entities.get("subjects_discussed", []))
        self.grades_discussed = set(entities.get("grades_discussed", []))
        self.pages_mentioned = set(entities.get("pages_mentioned", []))
        self.last_topic = entities.get("last_topic", "")
        self.last_example = entities.get("last_example")
        self.last_page = entities.get("last_page")
        
        analytics = data.get("analytics", {})
        self.query_types = analytics.get("query_types", [])
        
        self.episodes = data.get("episodes", [])
        
        concept_graph_data = data.get("concept_graph", {})
        self.concept_graph = {k: set(v) for k, v in concept_graph_data.items()}
