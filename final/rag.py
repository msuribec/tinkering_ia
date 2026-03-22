from __future__ import annotations

import numpy as np
import faiss
import google.generativeai as genai

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 768  # text-embedding-004 output size
NUMERIC_KEYWORDS = {"total", "sum", "how much", "spent", "cost", "amount", "price"}


# ── Text representation ───────────────────────────────────────────────────────

def receipt_to_text(receipt: dict) -> str:
    """Convert a receipt dict into a natural language summary for embedding."""
    data = receipt.get("data", {})
    vendor = data.get("vendor") or "Unknown vendor"
    date = data.get("date") or "Unknown date"
    currency = data.get("currency") or ""
    total = data.get("total") or 0
    items = data.get("items") or []
    category_totals = data.get("category_totals") or {}
    savings_tip = data.get("savings_tip") or ""

    lines = [f"Receipt from {vendor} on {date}."]

    if items:
        lines.append("Items:")
        for item in items:
            name = item.get("name") or "Unknown item"
            price = item.get("price") or 0
            category = item.get("category") or "Uncategorized"
            lines.append(f"- {name} ({category}) {price} {currency}")

    lines.append(f"Total: {total} {currency}")

    if category_totals:
        cats = ", ".join(f"{k}: {v} {currency}" for k, v in category_totals.items())
        lines.append(f"Spending by category: {cats}")

    if savings_tip:
        lines.append(f"Savings tip: {savings_tip}")

    return "\n".join(lines)


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """Embed text using Gemini text-embedding-004. Returns a 768-dim float list."""
    result = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


# ── Vector store ──────────────────────────────────────────────────────────────

class ReceiptVectorStore:
    """FAISS-backed vector store for receipt text embeddings."""

    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._receipts: list[dict] = []
        self._texts: list[str] = []

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add_receipt(self, receipt: dict) -> None:
        text = receipt_to_text(receipt)
        vec = np.array([embed_text(text, task_type="retrieval_document")], dtype="float32")
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._receipts.append(receipt)
        self._texts.append(text)

    def search(self, query: str, k: int = 5) -> list[str]:
        """Return top-k receipt text summaries most relevant to query."""
        if self._index.ntotal == 0:
            return []
        vec = np.array([embed_text(query, task_type="retrieval_query")], dtype="float32")
        faiss.normalize_L2(vec)
        actual_k = min(k, self._index.ntotal)
        _, indices = self._index.search(vec, actual_k)
        return [self._texts[i] for i in indices[0] if i >= 0]

    def _search_receipts(self, query: str, k: int = 5) -> list[dict]:
        """Return top-k raw receipt dicts most relevant to query (for numeric aggregation)."""
        if self._index.ntotal == 0:
            return []
        vec = np.array([embed_text(query, task_type="retrieval_query")], dtype="float32")
        faiss.normalize_L2(vec)
        actual_k = min(k, self._index.ntotal)
        _, indices = self._index.search(vec, actual_k)
        return [self._receipts[i] for i in indices[0] if i >= 0]


# ── RAG pipeline ──────────────────────────────────────────────────────────────

def answer_question(question: str, vector_store: ReceiptVectorStore, model) -> str:
    """
    Retrieve relevant receipts and answer the question with the LLM.

    For aggregate questions (total/sum/how much), also computes a Python-side
    numeric total and injects it into the prompt as a grounding anchor.
    """
    context_texts = vector_store.search(question, k=5)

    if not context_texts:
        return "No receipts found in your history yet. Please add some receipts first."

    # Hybrid: compute numeric total for aggregate questions
    hybrid_note = ""
    if any(kw in question.lower() for kw in NUMERIC_KEYWORDS):
        relevant_receipts = vector_store._search_receipts(question, k=5)
        computed_total = sum(
            float(r.get("data", {}).get("total") or 0) for r in relevant_receipts
        )
        currency = relevant_receipts[0].get("data", {}).get("currency", "") if relevant_receipts else ""
        hybrid_note = (
            f"\n[COMPUTED] Python sum of totals from the most relevant receipts: "
            f"{computed_total:.2f} {currency}\n"
        )

    context_block = "\n\n---\n\n".join(context_texts)

    prompt = (
        "You are a personal finance assistant. "
        "Answer the user's question using ONLY the receipt data provided below.\n"
        "If the answer requires calculation, compute it. "
        "If the answer cannot be determined from the data, say so clearly.\n\n"
        f"RECEIPTS:\n{context_block}\n"
        f"{hybrid_note}\n"
        f"QUESTION: {question}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()
