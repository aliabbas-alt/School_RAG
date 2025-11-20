# run_agent.py
"""
Enhanced Q&A system with advanced conversation memory and grade/subject awareness.
Multimodal RAG: Text + Vision-based image descriptions + Grade/Subject filtering.
Features: Entity tracking, smart context, query classification, automatic filtering.
"""

import os
import re
import json
from typing import List, Dict, Set, Optional
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from supabase_tool import search_school_documents
from math_search_tool import search_math_documents

# Load environment variables
load_dotenv()


class EnhancedConversationMemory:
    """
    Advanced conversation memory with topic tracking, entity recognition,
    and smart summarization for educational RAG systems.
    """
    
    def __init__(self, max_history: int = 15, max_context_tokens: int = 2000):
        """
        Initialize enhanced memory.
        
        Args:
            max_history: Maximum interactions to store
            max_context_tokens: Max tokens for context (approximate)
        """
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.history = []
        self.session_start = datetime.now()
        
        # Enhanced tracking
        self.topics_discussed: Set[str] = set()
        self.pages_mentioned: Set[int] = set()
        self.subjects_discussed: Set[str] = set()
        self.grades_discussed: Set[int] = set()  # NEW
        self.question_count = 0
        self.clarification_count = 0
    
    def extract_entities(self, text: str) -> Dict[str, List]:
        """Extract educational entities from text including grades."""
        entities = {
            "pages": [],
            "subjects": [],
            "grades": [],  # NEW
            "topics": [],
            "chapters": []
        }
        
        # Extract page numbers
        page_pattern = r'\bpage\s+(\d+)\b|\bp\.?\s*(\d+)\b'
        pages = re.findall(page_pattern, text.lower())
        for match in pages:
            page_num = int(match[0] or match[1])
            entities["pages"].append(page_num)
            self.pages_mentioned.add(page_num)
        
        # Extract grades (NEW)
        grade_pattern = r'\bgrade\s+(\d+)\b|\bclass\s+(\d+)\b'
        grades = re.findall(grade_pattern, text.lower())
        for match in grades:
            grade_num = int(match[0] or match[1])
            if 1 <= grade_num <= 12:
                entities["grades"].append(grade_num)
                self.grades_discussed.add(grade_num)
        
        # Extract subjects (basic keyword matching)
        subjects = ["math", "physics", "chemistry", "biology", "english", 
                   "history", "geography", "computer", "science"]
        for subject in subjects:
            if subject in text.lower():
                entities["subjects"].append(subject)
                self.subjects_discussed.add(subject)
        
        # Extract chapter numbers
        chapter_pattern = r'\bchapter\s+(\d+)\b|\bch\.?\s*(\d+)\b'
        chapters = re.findall(chapter_pattern, text.lower())
        for match in chapters:
            entities["chapters"].append(int(match[0] or match[1]))
        
        return entities
    
    def extract_math_query_entities(self, query: str) -> dict:
        """Extract math-specific metadata from a query."""
        query_lower = query.lower()
        entities = {}

        # Detect exercise numbers like "Exercise 7.2"
        ex_match = re.search(r"exercise\s+(\d+(?:\.\d+)?)", query_lower)
        if ex_match:
            entities["exercise_number"] = ex_match.group(1)

        # Detect example numbers like "Example 5"
        exm_match = re.search(r"example\s+(\d+)", query_lower)
        if exm_match:
            entities["example_number"] = exm_match.group(1)

        # Detect question numbers like "Question 2"
        q_match = re.search(r"question\s+(\d+)", query_lower)
        if q_match:
            entities["question_number"] = q_match.group(1)

        # Detect sub-questions like "(ii)"
        sub_match = re.search(r"\(\s*([ivx]+)\s*\)", query_lower)
        if sub_match:
            entities["sub_question"] = sub_match.group(1).lower()

        # Detect subject
        if "math" in query_lower or "mathematics" in query_lower:
            entities["subject"] = "Math"

        return entities
    
    def classify_query_type(self, query: str) -> str:
        """Classify the type of user query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "explanation"
        elif any(word in query_lower for word in ["how", "steps", "process"]):
            return "procedural"
        elif any(word in query_lower for word in ["image", "diagram", "picture", "figure"]):
            return "visual"
        elif any(word in query_lower for word in ["list", "all", "show me"]):
            return "aggregation"
        elif query_lower.startswith(("can", "could", "would")):
            self.clarification_count += 1
            return "clarification"
        else:
            return "general"
    
    def add_interaction(
        self, 
        user_query: str, 
        assistant_response: str,
        search_results_count: int = 0
    ):
        """Add interaction with enhanced metadata."""
        
        # Extract entities
        query_entities = self.extract_entities(user_query)
        response_entities = self.extract_entities(assistant_response)
        
        # Classify query
        query_type = self.classify_query_type(user_query)
        
        # Update topic tracking
        if query_entities["subjects"]:
            self.topics_discussed.update(query_entities["subjects"])
        
        # Create interaction record
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "assistant": assistant_response,
            "query_type": query_type,
            "entities": query_entities,
            "search_results": search_results_count,
            "turn_number": len(self.history) + 1
        }
        
        self.history.append(interaction)
        self.question_count += 1
        
        # Trim if exceeds max
        if len(self.history) > self.max_history:
            # Keep first and recent interactions
            self.history = [self.history[0]] + self.history[-(self.max_history-1):]
    
    def get_context(self, last_n: int = 3, prioritize_recent: bool = True) -> str:
        """Get conversation context with smart filtering."""
        if not self.history:
            return "No previous conversation."
        
        # Get relevant interactions
        if prioritize_recent:
            relevant = self.history[-last_n:]
        else:
            relevant = self.history[:last_n]
        
        context_parts = ["=== Conversation Context ==="]
        
        # Add session metadata
        if self.pages_mentioned:
            pages_str = ", ".join(map(str, sorted(self.pages_mentioned)[-5:]))
            context_parts.append(f"Pages discussed: {pages_str}")
        
        if self.grades_discussed:  # NEW
            context_parts.append(f"Grades: {', '.join(map(str, sorted(self.grades_discussed)))}")
        
        if self.subjects_discussed:
            context_parts.append(f"Subjects: {', '.join(self.subjects_discussed)}")
        
        context_parts.append("")
        
        # Add recent interactions
        for i, interaction in enumerate(relevant, 1):
            turn = interaction['turn_number']
            query_type = interaction.get('query_type', 'general')
            
            context_parts.append(f"Turn {turn} [{query_type}]:")
            context_parts.append(f"User: {interaction['user']}")
            
            # Summarize long responses
            response = interaction['assistant']
            if len(response) > 200:
                response = response[:200] + "... [continued]"
            context_parts.append(f"Assistant: {response}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_relevant_entities(self) -> str:
        """Get summary of entities discussed."""
        entities_summary = []
        
        if self.pages_mentioned:
            pages = sorted(self.pages_mentioned)
            if len(pages) <= 5:
                entities_summary.append(f"Pages: {', '.join(map(str, pages))}")
            else:
                entities_summary.append(f"Pages: {pages[0]}-{pages[-1]} (and {len(pages)-2} others)")
        
        if self.grades_discussed:  # NEW
            grades_str = ', '.join(map(str, sorted(self.grades_discussed)))
            entities_summary.append(f"Grades: {grades_str}")
        
        if self.subjects_discussed:
            entities_summary.append(f"Subjects: {', '.join(self.subjects_discussed)}")
        
        return " | ".join(entities_summary) if entities_summary else "No specific entities tracked"
    
    def get_summary(self) -> str:
        """Enhanced conversation summary."""
        if not self.history:
            return "No conversation yet."
        
        duration = (datetime.now() - self.session_start).seconds // 60
        
        # Count query types
        query_types = Counter(i.get('query_type', 'general') for i in self.history)
        most_common_type = query_types.most_common(1)[0] if query_types else ("general", 0)
        
        return f"""📊 Conversation Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Session Duration: {duration} minutes
Total Interactions: {len(self.history)}
Questions Asked: {self.question_count}
Clarifications: {self.clarification_count}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Most Common Query Type: {most_common_type[0]} ({most_common_type[1]} times)
{self.get_relevant_entities()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First Query: {self.history[0]['user'][:50]}...
Last Query: {self.history[-1]['user'][:50]}..."""
    
    def get_smart_context_for_query(self, current_query: str) -> str:
        """
        Generate context specifically relevant to current query.
        Uses entity overlap to prioritize relevant history.
        """
        if not self.history:
            return self.get_context(last_n=2)
        
        # Extract entities from current query
        current_entities = self.extract_entities(current_query)
        
        # Find relevant past interactions
        relevant_interactions = []
        
        for interaction in reversed(self.history[-10:]):  # Check last 10
            past_entities = interaction.get('entities', {})
            
            # Check for entity overlap
            page_overlap = bool(set(current_entities['pages']) & set(past_entities.get('pages', [])))
            subject_overlap = bool(set(current_entities['subjects']) & set(past_entities.get('subjects', [])))
            grade_overlap = bool(set(current_entities['grades']) & set(past_entities.get('grades', [])))  # NEW
            
            if page_overlap or subject_overlap or grade_overlap:
                relevant_interactions.append(interaction)
            
            if len(relevant_interactions) >= 3:
                break
        
        # If no relevant history, use recent
        if not relevant_interactions:
            return self.get_context(last_n=2)
        
        # Format relevant context
        context_parts = ["=== Relevant Context ==="]
        context_parts.append(self.get_relevant_entities())
        context_parts.append("")
        
        for interaction in relevant_interactions:
            turn = interaction['turn_number']
            context_parts.append(f"Turn {turn}:")
            context_parts.append(f"Q: {interaction['user']}")
            context_parts.append(f"A: {interaction['assistant'][:150]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Reset memory."""
        self.history = []
        self.topics_discussed.clear()
        self.pages_mentioned.clear()
        self.subjects_discussed.clear()
        self.grades_discussed.clear()  # NEW
        self.question_count = 0
        self.clarification_count = 0
        self.session_start = datetime.now()
    
    def save_session(self, filepath: str):
        """Save session with metadata."""
        session_data = {
            "metadata": {
                "session_start": self.session_start.isoformat(),
                "session_end": datetime.now().isoformat(),
                "total_interactions": len(self.history),
                "topics_discussed": list(self.topics_discussed),
                "pages_mentioned": sorted(list(self.pages_mentioned)),
                "subjects_discussed": list(self.subjects_discussed),
                "grades_discussed": sorted(list(self.grades_discussed))  # NEW
            },
            "history": self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Session saved to {filepath}")


def create_qa_system():
    """Create Q&A system with LLM and enhanced system prompt."""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    system_prompt = """You are an intelligent educational assistant for a school management system.

You have access to a comprehensive database containing:
- Textbook content organized by GRADE (1-12) and SUBJECT
- Text AND detailed AI-generated image descriptions
- School policies, handbooks, and guidelines
- Curriculum materials (CBSE, SSE, ICSE, IB, etc.)
- Student and parent information
- Academic procedures and rules

**Grade & Subject Organization:**
The database is automatically organized and filtered by:
- Grade levels (1-12) - detected from queries like "Grade 8" or "Class 10"
- Subjects (Math, Physics, Chemistry, Biology, English, etc.) - detected from keywords
- You don't need to explicitly ask for filtering - it happens automatically

**IMPORTANT - Image Descriptions:**
The database includes AI-generated descriptions of ALL images, diagrams, and visual elements from textbooks. These descriptions are extremely detailed, including:
- Complete descriptions of all images on each page
- Colors, positions, and arrangements
- Visual elements like diagrams, charts, activities
- Educational context and purpose

**Conversation Context:**
You have access to recent conversation history with entity tracking (pages, grades, subjects). Use it to:
- Understand follow-up questions (e.g., "What about the next page?")
- Resolve pronouns (e.g., "Tell me more about that")
- Maintain conversational flow across topics
- Recognize when user returns to previously discussed topics
- Remember grade/subject context from earlier in conversation

**Automatic Search Features:**
When users ask questions, the system automatically:
- Detects grade level from natural language
- Identifies subject from keywords
- Filters results to relevant grade/subject
- Prioritizes image descriptions for visual queries
- Returns grade-appropriate content

**Example Queries You Can Handle:**
- "What is photosynthesis?" → Searches science content
- "Grade 8 math chapter 3" → Filters to Grade 8 Math only
- "Physics Grade 10 circuit diagrams" → Grade 10 Physics images
- "Show me the images on page 16" → Retrieves page-specific content
- "Chemistry equations" → Chemistry content with automatic filtering

When answering questions:
1. First search the database (automatic grade/subject filtering happens)
2. Pay special attention to IMAGE_DESCRIPTION results - they contain complete visual information
3. For image questions, provide COMPLETE descriptions from the database
4. Use conversation history and entity context to understand follow-ups
5. Always cite sources with [Source: filename, Page: number, Grade: X, Subject: Y]
6. If information is not found, say so clearly
7. Be conversational, clear, and educational
8. Mention grade/subject context when relevant to the answer

Remember: The database contains detailed descriptions of EVERY image in textbooks across all grades and subjects!"""

    return llm, system_prompt


def main():
    """Main interactive loop with enhanced memory and grade/subject awareness."""
    
    print("="*70)
    print("  🎓 SCHOOL DOCUMENT Q&A SYSTEM")
    print("="*70)
    print("Enhanced Multimodal RAG with Smart Memory + Grade/Subject Filtering")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Features:")
    print("  ✓ Text + Image descriptions from textbooks")
    print("  ✓ Automatic Grade & Subject detection and filtering")
    print("  ✓ Entity tracking (pages, subjects, chapters, grades)")
    print("  ✓ Smart context-aware responses")
    print("  ✓ Query classification and analytics")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\nCommands:")
    print("  'exit', 'quit', 'q' → Exit system")
    print("  'memory'           → Show conversation analytics")
    print("  'clear'            → Reset conversation memory")
    print("  'save'             → Save session to file")
    print("\n" + "="*70)
    
    try:
        llm, system_prompt = create_qa_system()
        memory = EnhancedConversationMemory(max_history=15)
        print("✅ System initialized successfully!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nCheck your .env file has OPENAI_API_KEY")
        return
    
    while True:
        try:
            # Get user input
            query = input("\n💬 You: ").strip()
                          
            if not query:
                continue

            # Handle commands
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                print(f"\n{memory.get_summary()}")
                
                # Ask to save
                save_prompt = input("\n💾 Save session? (y/n): ").strip().lower()
                if save_prompt == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"session_{timestamp}.json"
                    memory.save_session(filename)
                break
            
            if query.lower() == "memory":
                print("\n" + "="*70)
                print(memory.get_summary())
                print("="*70)
                print("\n" + memory.get_context(last_n=5))
                print("="*70)
                continue
            
            if query.lower() == "clear":
                memory.clear()
                print("✅ Conversation memory cleared!")
                continue
            
            if query.lower() == "save":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_{timestamp}.json"
                memory.save_session(filename)
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
            print(f"\n{memory.get_summary()}")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
