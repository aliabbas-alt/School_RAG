# memory/conversation_memory.py
"""
Conversation memory for multi-turn dialogues in RAG system.
Stores chat history for context-aware responses.
"""

from typing import List, Dict, Any
from datetime import datetime
import json
import os


class ConversationMemory:
    """Manages conversation history for context-aware RAG."""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to remember
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
    
    def add_interaction(
        self, 
        user_query: str, 
        assistant_response: str,
        retrieved_docs: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add a user-assistant interaction to memory.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's answer
            retrieved_docs: Documents retrieved from RAG (optional)
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "assistant": assistant_response,
            "docs_retrieved": len(retrieved_docs) if retrieved_docs else 0
        }
        
        self.history.append(interaction)
        self.metadata["total_queries"] += 1
        
        # Keep only last N interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_conversation_context(self, last_n: int = 3) -> str:
        """
        Get recent conversation history as formatted string.
        
        Args:
            last_n: Number of recent interactions to include
            
        Returns:
            Formatted conversation history
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
    
    def get_last_query(self) -> str:
        """Get the user's last query."""
        if not self.history:
            return ""
        return self.history[-1]["user"]
    
    def get_last_response(self) -> str:
        """Get the assistant's last response."""
        if not self.history:
            return ""
        return self.history[-1]["assistant"]
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save conversation history to JSON file."""
        data = {
            "metadata": self.metadata,
            "history": self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_file(self, filepath: str) -> None:
        """Load conversation history from JSON file."""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        self.history = data.get("history", [])
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.history:
            return "No conversation yet."
        
        topics = set()
        for interaction in self.history:
            query = interaction["user"].lower()
            if "page" in query:
                topics.add("page-specific queries")
            if "image" in query or "diagram" in query:
                topics.add("visual content")
            if "explain" in query:
                topics.add("explanations")
        
        return f"""Conversation Summary:
- Total queries: {self.metadata['total_queries']}
- Topics discussed: {', '.join(topics) if topics else 'general'}
- Session duration: {len(self.history)} turns"""
