import re

def chunk_math_semantic(text: str, page: int, pdf_source: str) -> list[dict]:
    chunks = []
    current_chunk = []
    chunk_type = "text"

    # Track context
    example_number = None
    exercise_number = None
    question_number = None
    sub_question = None
    linked_example_id = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # --- Detect Example ---
        ex_match = re.match(r"^Example\s+(\d+)", line, re.I)
        if ex_match:
            # Save previous chunk
            if current_chunk:
                chunks.append({
                    "page_content": "\n".join(current_chunk),
                    "metadata": {
                        "page": page,
                        "pdf_source": pdf_source,
                        "chunk_type": chunk_type,
                        "example_number": example_number,
                        "exercise_number": exercise_number,
                        "question_number": question_number,
                        "sub_question": sub_question,
                        "linked_example_id": linked_example_id
                    }
                })
                current_chunk = []
            chunk_type = "example"
            example_number = ex_match.group(1)
            exercise_number = None
            question_number = None
            sub_question = None
            linked_example_id = None

        # --- Detect Exercise ---
        exr_match = re.match(r"^Exercise\s+(\d+(?:\.\d+)?)", line, re.I)
        if exr_match:
            if current_chunk:
                chunks.append({
                    "page_content": "\n".join(current_chunk),
                    "metadata": {
                        "page": page,
                        "pdf_source": pdf_source,
                        "chunk_type": chunk_type,
                        "example_number": example_number,
                        "exercise_number": exercise_number,
                        "question_number": question_number,
                        "sub_question": sub_question
                    }
                })
                current_chunk = []
            chunk_type = "exercise"
            exercise_number = exr_match.group(1)
            example_number = None
            question_number = None
            sub_question = None

        # --- Detect Solution ---
        elif line.lower().startswith("solution"):
            if current_chunk:
                chunks.append({
                    "page_content": "\n".join(current_chunk),
                    "metadata": {
                        "page": page,
                        "pdf_source": pdf_source,
                        "chunk_type": chunk_type,
                        "example_number": example_number,
                        "exercise_number": exercise_number,
                        "question_number": question_number,
                        "sub_question": sub_question
                    }
                })
                current_chunk = []
            chunk_type = "solution"
            linked_example_id = example_number  # link back to example

        # --- Detect Question Numbers inside Exercises ---
        q_match = re.match(r"^(\d+)[\.\)]", line)
        if q_match and chunk_type == "exercise":
            if current_chunk:
                chunks.append({
                    "page_content": "\n".join(current_chunk),
                    "metadata": {
                        "page": page,
                        "pdf_source": pdf_source,
                        "chunk_type": chunk_type,
                        "exercise_number": exercise_number,
                        "question_number": question_number,
                        "sub_question": sub_question
                    }
                })
                current_chunk = []
            question_number = q_match.group(1)
            sub_question = None

        # --- Detect Sub-questions (i), (ii), (iii) ---
        sub_match = re.match(r"^\(\s*(i+)\s*\)", line, re.I)
        if sub_match and chunk_type in ["exercise", "example"]:
            if current_chunk:
                chunks.append({
                    "page_content": "\n".join(current_chunk),
                    "metadata": {
                        "page": page,
                        "pdf_source": pdf_source,
                        "chunk_type": chunk_type,
                        "exercise_number": exercise_number,
                        "example_number": example_number,
                        "question_number": question_number,
                        "sub_question": sub_question
                    }
                })
                current_chunk = []
            sub_question = sub_match.group(1)

        # Add line to current chunk
        current_chunk.append(line)

    # --- Save last chunk ---
    if current_chunk:
        chunks.append({
            "page_content": "\n".join(current_chunk),
            "metadata": {
                "page": page,
                "pdf_source": pdf_source,
                "chunk_type": chunk_type,
                "example_number": example_number,
                "exercise_number": exercise_number,
                "question_number": question_number,
                "sub_question": sub_question,
                "linked_example_id": linked_example_id
            }
        })

    return chunks