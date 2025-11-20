import re
from supabase_tool import SmartSearchEngine
from langchain_core.tools import Tool
from embeddings.registry import build_provider

class MathSmartSearchEngine(SmartSearchEngine):
    """Custom search engine specialized for math queries with LaTeX prioritization and metadata search."""

    def __init__(self):
        super().__init__()
        self.math_keywords = [
            "integral", "derivative", "equation", "function", "limit", "matrix",
            "theorem", "proof", "latex", "symbol", "expression", "solve", "graph", "math"
        ]

    def detect_query_intent(self, query: str) -> dict:
        """Override intent detection to bias toward math topics and extract metadata filters."""
        intent = super().detect_query_intent(query)

        q_lower = query.lower()

        # Force subject to Math if keywords match
        if any(kw in q_lower for kw in self.math_keywords):
            intent["subject"] = "Math"

        # Example number detection
        example_match = re.search(r'example\s+(\d+)', q_lower)
        if example_match:
            intent["example_number"] = example_match.group(1)

        # Exercise number detection (supports 7 or 7.2 style)
        exercise_match = re.search(r'exercise\s+(\d+(?:\.\d+)?)', q_lower)
        if exercise_match:
            intent["exercise_number"] = exercise_match.group(1)

        # Question number detection
        question_match = re.search(r'question\s+(\d+)', q_lower)
        if question_match:
            intent["question_number"] = question_match.group(1)

        # Sub-question detection (i, ii, iii, etc.)
        sub_match = re.search(r'\(\s*([ivx]+)\s*\)', q_lower)
        if sub_match:
            intent["sub_question"] = sub_match.group(1)

        return intent

    def filter_math_results(self, results: list[dict]) -> list[dict]:
        """Keep only math-related chunks (text or image)."""
        return [r for r in results if r.get("content_type") in ["math_text", "math_image", "exercise", "example", "solution"]]

    def keyword_search_math(self, intent: dict) -> list[dict]:
        """
        Direct keyword/metadata search scoped to Math.
        Handles Example numbers, Exercise numbers, Question numbers, Sub-questions.
        """
        try:
            query = self.vector_store.client.table("math_documents").select("*")

            # Example search
            if intent.get("example_number"):
                query = query.eq("metadata->>example_number", str(intent["example_number"]))
                print(f"   🔍 Filtering: metadata->>example_number = {intent['example_number']}")

            if intent.get("exercise_number") is not None:
                query = query.eq("metadata->>exercise_number", str(intent["exercise_number"]))
                print(f"   🔍 Filtering: metadata->>exercise_number = {intent['exercise_number']}")

            if intent.get("question_number") is not None:
                query = query.eq("metadata->>question_number", str(intent["question_number"]))
                print(f"   🔍 Filtering: metadata->>question_number = {intent['question_number']}")

            if intent.get("sub_question") is not None:
                query = query.eq("metadata->>sub_question", str(intent["sub_question"]))
                print(f"   🔍 Filtering: metadata->>sub_question = {intent['sub_question']}")

            response = query.limit(20).execute()
            if not (hasattr(response, "data") and response.data):
                return []

            results = response.data

            # Prioritize math_image > math_text > other
            vision_results = [r for r in results if r.get("content_type") == "math_image"]
            text_results = [r for r in results if r.get("content_type") == "math_text"]
            other_results = [r for r in results if r.get("content_type") not in ["math_image", "math_text"]]

            vision_results.sort(key=lambda x: x.get("page", 999))
            text_results.sort(key=lambda x: x.get("page", 999))
            other_results.sort(key=lambda x: x.get("page", 999))

            all_results = vision_results + text_results + other_results

            return all_results[:10]

        except Exception as e:
            print(f"   ⚠️ Math keyword search error: {e}")
            return []

    def search_with_intent(self, query: str, intent: dict) -> list[dict]:
        """
        Hybrid search: semantic first, then enrich with metadata matches.
        Returns combined results with chunk IDs.
        """

        combined_results = []
        seen_ids = set()

        # Step 1: Semantic search first
        try:
            # Use large embedding model for Math
            provider = build_provider("openai", model="text-embedding-3-large")
            print("   🔮 Using text-embedding-3-large for Math query")

            query_result = provider.embed_texts([query])
            query_embedding = query_result.vectors[0]

            semantic_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                match_threshold=0.37,
                match_count=40,
                grade=intent.get("grade"),
                subject="Math"
            )

            semantic_results = self.filter_math_results(semantic_results)

            for r in semantic_results:
                chunk_id = r.get("id") or r.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_results.append(r)

            print(f"   ✅ Found {len(semantic_results)} semantic results")

        except Exception as e:
            print(f"   ⚠️ Semantic search error: {e}")

        # Step 2: Enrich with metadata matches
        try:
            keyword_results = self.keyword_search_math(intent)
            for r in keyword_results:
                chunk_id = r.get("id") or r.get("chunk_id")
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    combined_results.append(r)

            if keyword_results:
                print(f"   ✅ Added {len(keyword_results)} metadata results")

        except Exception as e:
            print(f"   ⚠️ Metadata search error: {e}")

        # Step 3: Sort combined results
        vision_results = [r for r in combined_results if r.get("content_type") == "math_image"]
        text_results = [r for r in combined_results if r.get("content_type") == "math_text"]
        other_results = [r for r in combined_results if r.get("content_type") not in ["math_image", "math_text"]]

        vision_results.sort(key=lambda x: x.get("page", 999))
        text_results.sort(key=lambda x: x.get("page", 999))
        other_results.sort(key=lambda x: x.get("page", 999))

        final_results = vision_results + text_results + other_results

        return final_results[:15]

    # def search_with_intent(self, query: str, intent: dict) -> list[dict]:
    #     """Override search to prioritize math content with metadata first, then semantic ranking."""
    #     # Step 1: Try keyword/metadata search
    #     keyword_results = self.keyword_search_math(intent)
    #     if keyword_results:
    #         print(f"   ✅ Found {len(keyword_results)} results via math keyword/metadata search")
    #         return keyword_results

    #     # Step 2: Fall back to semantic search
    #     query_result = self.provider.embed_texts([query])
    #     query_embedding = query_result.vectors[0]  # flatten

    #     results = self.vector_store.similarity_search(
    #         query_embedding=query_embedding,
    #         match_threshold=0.22,
    #         match_count=40,
    #         grade=intent.get("grade"),
    #         subject="Math"
    #     )

    #     return self.filter_math_results(results)


_math_engine = None

def get_math_engine() -> MathSmartSearchEngine:
    global _math_engine
    if _math_engine is None:
        _math_engine = MathSmartSearchEngine()
    return _math_engine

def search_math_documents(query: str, memory_context: dict | None = None) -> str:
    try:
        engine = get_math_engine()
        return engine.search(query, memory_context=memory_context, verbose=True)
    except Exception as e:
        import traceback
        return f"❌ Math Search Error: {e}\n\nTraceback:\n{traceback.format_exc()}"

math_search_tool = Tool.from_function(
    name="search_math_documents",
    description="""Search math-specific educational content with LaTeX prioritization, example detection, and semantic filters.

Use for:
- "Example 8 on integrals"
- "Show me derivative graph"
- "Explain this equation"
- "Grade 11 calculus problems"
""",
    func=search_math_documents,
    return_direct=False
)