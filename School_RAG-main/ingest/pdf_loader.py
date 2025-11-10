from __future__ import annotations
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
import os


def load_pdf(path: str, use_unstructured: bool = False) -> List[Document]:
    """
    Load a PDF into LangChain Document objects.
    - PyPDFLoader: good for digital PDFs; keeps page metadata.
    - UnstructuredPDFLoader: better for complex or scanned PDFs.

    Returns a list of Documents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    if use_unstructured:
        loader = UnstructuredPDFLoader(path)
        try:
            docs = loader.load()
        except ModuleNotFoundError as e:
            if "pdfminer" in str(e).lower():
                raise ModuleNotFoundError(
                    "UnstructuredPDFLoader requires 'pdfminer.six' or 'unstructured[pdf]'. "
                    "Install with: pip install pdfminer.six  OR  pip install \"unstructured[pdf]\""
                )
            raise
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        return docs

    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", os.path.basename(path))
    return docs