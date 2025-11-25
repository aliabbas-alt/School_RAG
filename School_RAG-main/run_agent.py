# run_agent.py
"""
Production-Ready School RAG Agent for Indian School Management System

Features:
- Advanced conversation memory with attention mechanism & temporal weighting
- School data isolation for multi-tenancy
- Grade/Subject automatic filtering
- Hybrid search (keyword + semantic with RRF)
- Episode tracking (study sessions)
- Session persistence and analytics
- Comprehensive error handling
- Works with LangChain 1.0+

Supports: CBSE, SSE, ICSE, IB, and other Indian curricula
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from supabase_tool import search_school_documents, get_search_engine
from memory.conversation_memory import EnhancedConversationMemory  # ← USES NEW ADVANCED MEMORY

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================================================
# SCHOOL RAG AGENT
# ============================================================================

class SchoolRAGAgent:
    """
    Production-ready RAG agent for Indian School Management System.
    
    Features:
    - Advanced conversation memory with attention mechanism
    - Temporal weighting (recent + relevant)
    - Episode tracking (study sessions)
    - Multi-hop concept reasoning
    - School data isolation
    - Grade/Subject automatic filtering
    - Session persistence
    - Comprehensive error handling
    
    Supports: CBSE, SSE, ICSE, IB, and other curricula
    """
    
    def __init__(
        self,
        school_id: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2
    ):
        """
        Initialize School RAG Agent.
        
        Args:
            school_id: School ID for data isolation (REQUIRED)
            model: LLM model to use
            temperature: LLM temperature (0-1)
        """
        self.school_id = school_id
        self.memory = EnhancedConversationMemory(max_history=15)  # ← NEW ADVANCED MEMORY
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # System prompt
        self.system_prompt = """You are an intelligent educational assistant for an Indian school management system.
You help students, teachers, and administrators with questions about textbooks, curriculum, policies, and procedures.

**Database Contents:**
- Textbook content organized by GRADE (1-12) and SUBJECT
- Complete text + AI-generated image descriptions
- School policies, handbooks, and procedures
- Curriculum materials (CBSE, SSE, ICSE, IB, etc.)

**Automatic Features:**
- Grade/Subject detection and filtering from queries
- Hybrid search (Keyword + Semantic with RRF)
- Vision-extracted math prioritized over OCR
- Memory-based query enhancement with attention mechanism
- Temporal weighting (recent + relevant prioritization)

**Instructions:**
1. Use search results to provide accurate answers
2. Always cite sources: [Source: filename, Page: X, Grade: Y, Subject: Z]
3. Use conversation history to understand follow-up questions
4. For vague queries like "example 8" or "next problem", use memory context
5. If information not found, say so clearly - don't fabricate
6. For curriculum questions (CBSE/SSE/ICSE/IB), respect the curriculum type
7. Be educational, clear, and helpful"""
    
    def run(self, query: str) -> str:
        """
        Run complete RAG pipeline with advanced memory.
        
        Args:
            query: User's question
        
        Returns:
            Generated answer with citations
        """
        try:
            logger.info(f"Processing query: {query}")
            logger.info(f"School: {self.school_id}")
            
            # Get memory contexts
            memory_dict = self.memory.get_context_for_enhancement()
            
            # ← CHANGED: Use attention-weighted context for best performance
            memory_string = self.memory.get_attention_weighted_context(query, top_k=3)
            
            # Alternative options (uncomment to use):
            # memory_string = self.memory.get_temporal_weighted_context(query, decay_rate=0.1)
            # memory_string = self.memory.get_smart_context(query)
            
            logger.info("Searching database...")
            
            # Search documents with memory context and school isolation
            search_results = search_school_documents(
                query,
                memory_context=memory_dict,
                school_id=self.school_id
            )
            
            logger.info("Generating response...")
            
            # Create user message with context
            user_content = f"""**User Question:** {query}

**Conversation Context:**
{memory_string}

**Search Results:**
{search_results}

Based on the search results and conversation context, provide a clear and accurate answer.
Always cite your sources with [Source: filename, Page: X, Grade: Y]."""
            
            # Generate response
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_content)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Update memory
            self.memory.add_interaction(query, answer)
            
            logger.info("Response generated successfully")
            
            return answer
        
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            return f"❌ I encountered an error while processing your question: {str(e)}\n\nPlease try rephrasing or contact support."
    
    def ask(self, query: str) -> str:
        """Alias for run() - more intuitive for API usage."""
        return self.run(query)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including memory analytics."""
        engine = get_search_engine()
        search_metrics = engine.get_metrics()
        
        return {
            "search_metrics": search_metrics,
            "conversation_metrics": {
                "total_interactions": len(self.memory.history),
                "questions_asked": self.memory.question_count,
                "clarifications": self.memory.clarification_count,
                "subjects_discussed": list(self.memory.subjects_discussed),
                "grades_discussed": sorted(list(self.memory.grades_discussed)),
                "episodes": len(self.memory.episodes),
                "pages_referenced": len(self.memory.pages_mentioned)
            }
        }
    
    def get_memory_summary(self) -> str:
        """Get conversation summary with analytics."""
        return self.memory.get_summary()
    
    def get_episode_summary(self) -> str:
        """Get study session summary."""
        return self.memory.get_episode_summary()
    
    def get_related_concepts(self, concept: str) -> set:
        """Get related concepts through multi-hop reasoning."""
        return self.memory.get_related_concepts(concept, max_depth=2)
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def save_session(self, filepath: str = None) -> str:
        """
        Save conversation session with all metadata.
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            Path to saved file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"sessions/session_{self.school_id}_{timestamp}.json"
        
        return self.memory.save_to_file(filepath)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def run_cli():
    """Interactive CLI interface with excellent UX."""
    print("="*70)
    print("  🎓 SCHOOL DOCUMENT Q&A SYSTEM")
    print("="*70)
    print("Production RAG with Advanced Memory + Hybrid Search")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Features:")
    print("  ✓ Attention-based memory retrieval (DMN-inspired)")
    print("  ✓ Temporal weighting (recent + relevant)")
    print("  ✓ Episode tracking (automatic study session detection)")
    print("  ✓ Multi-hop concept reasoning")
    print("  ✓ Automatic Grade & Subject detection")
    print("  ✓ Hybrid search (Keyword + Semantic with RRF)")
    print("  ✓ School data isolation for multi-tenancy")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\nCommands:")
    print("  'exit', 'quit', 'q' → Exit system")
    print("  'memory'           → Show conversation analytics")
    print("  'episodes'         → Show study sessions")
    print("  'metrics'          → Show performance metrics")
    print("  'clear'            → Reset conversation memory")
    print("  'save'             → Save session to file")
    print("\n" + "="*70)
    
    # Get school ID
    school_id = input("\n🏫 Enter School ID (or press Enter for demo): ").strip()
    if not school_id:
        school_id = "demo_school"
        print(f"   Using demo school: {school_id}")
    
    # Initialize agent
    try:
        agent = SchoolRAGAgent(school_id=school_id)
        print(f"\n✅ System initialized successfully for school: {school_id}\n")
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        print("   Check your .env file has OPENAI_API_KEY")
        return
    
    # Main loop
    while True:
        try:
            # Get user input
            query = input("\n💬 You: ").strip()
                          
            if not query:
                continue

            # Handle commands
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                print(f"\n{agent.get_memory_summary()}")
                
                # Ask to save
                save_prompt = input("\n💾 Save session? (y/n): ").strip().lower()
                if save_prompt == 'y':
                    filepath = agent.save_session()
                    print(f"✅ Session saved to: {filepath}")
                break
            
            if query.lower() == "memory":
                print("\n" + "="*70)
                print(agent.get_memory_summary())
                print("="*70)
                continue
            
            if query.lower() == "episodes":
                print("\n" + "="*70)
                print(agent.get_episode_summary())
                print("="*70)
                continue
            
            if query.lower() == "metrics":
                print("\n" + "="*70)
                print("📊 PERFORMANCE METRICS")
                print("="*70)
                metrics = agent.get_metrics()
                print(json.dumps(metrics, indent=2))
                print("="*70)
                continue
            
            if query.lower() == "clear":
                agent.clear_memory()
                print("✅ Conversation memory cleared!")
                continue
            
            if query.lower() == "save":
                filepath = agent.save_session()
                print(f"✅ Session saved to: {filepath}")
                continue

            # Step 1: Extract entities from query
            entities = memory.extract_math_query_entities(query)

            # Step 2: Decide routing based on metadata
            use_math_tool = False
            if entities.get("subject") == "Math":
                use_math_tool = True
            elif entities.get("exercise_number") or entities.get("example_number") or entities.get("question_number"):
                use_math_tool = True

            # Step 3: Route to correct search engine
            if use_math_tool:
                print("Using MathSmartSearchEngine (metadata-driven)")
                search_results = search_math_documents(query, memory_context={
                    "subjects_discussed": memory.subjects_discussed,
                    "grades_discussed": memory.grades_discussed,
                    "pages_mentioned": memory.pages_mentioned,
                    "last_topic": list(memory.topics_discussed)[-1] if memory.topics_discussed else ""
                })
            else:
                print("Using General SmartSearchEngine")
                search_results = search_school_documents(query, memory_context={
                    "subjects_discussed": memory.subjects_discussed,
                    "grades_discussed": memory.grades_discussed,
                    "pages_mentioned": memory.pages_mentioned,
                    "last_topic": list(memory.topics_discussed)[-1] if memory.topics_discussed else ""
                })
            
            # Process query
            print("\n🔍 Processing your question...\n")
            
            answer = agent.run(query)
            print("\n🔍 Searching database...")
            
            # Step 1: Search database (with automatic grade/subject filtering)
            # search_results = search_school_documents(query)
            # Detect if query is math-related
            query_lower = query.lower()
            math_keywords = [
                "integral", "derivative", "equation", "function", "limit", "matrix",
                "theorem", "proof", "latex", "symbol", "expression", "solve", "graph", "math"
            ]

            if any(kw in query.lower() for kw in math_keywords):
                print("Using MathSmartSearchEngine")
                search_results = search_math_documents(query, memory_context={
                    "subjects_discussed": memory.subjects_discussed,
                    "grades_discussed": memory.grades_discussed,
                    "pages_mentioned": memory.pages_mentioned,
                    "last_topic": list(memory.topics_discussed)[-1] if memory.topics_discussed else ""
                })
            else:
                print("Using General SmartSearchEngine")
                search_results = search_school_documents(query, memory_context={...})
        
            # Step 2: Build context with smart memory
            conversation_context = memory.get_smart_context_for_query(query)
            
            context = f"""User Question: {query}

{conversation_context}

Database Search Results:
{search_results}

Based on the search results above and the conversation context, provide a clear and accurate answer.
- For image questions, provide COMPLETE descriptions from IMAGE_DESCRIPTION results
- Use conversation history and entity tracking to understand follow-up questions
- If user asks about "that page" or "next page", use entity context
- Mention grade/subject when relevant to the answer
- Always cite sources with [Source: filename, Page: X]"""

            # Step 3: Get LLM response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            print("🤔 Generating answer...\n")
            
            response = llm.invoke(messages)
            answer = response.content
            
            # Store in memory with metadata
            memory.add_interaction(query, answer, search_results_count=30)
            
            # Display answer
            print("="*70)
            print("🤖 Assistant:")
            print("="*70)
            print(answer)
            print("="*70)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            print(f"\n{agent.get_memory_summary()}")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error("CLI error", exc_info=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    run_cli()


if __name__ == "__main__":
    main()
