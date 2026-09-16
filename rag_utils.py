import re
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded internal policy document.
# In a production system this would live in a real vector DB (e.g. Pinecone,
# Weaviate, pgvector) populated from a knowledge base. Here it's a static
# string to keep the demo self-contained, chunked into paragraph-sized
# passages so retrieval returns focused, citable context.
# ---------------------------------------------------------------------------
INTERNAL_POLICY_DOCUMENT = """
CANCELLATION POLICY: Customers may cancel any subscription within 30 days of
purchase for a full refund, no questions asked. After 30 days, cancellations
are prorated to the nearest billing cycle. Enterprise contracts with an
annual commitment require 60 days written notice and are subject to an early
termination fee equal to 20% of the remaining contract value. Cancellation
requests must be routed to the Billing team and logged with a risk score
reflecting churn likelihood.

BILLING DISPUTE POLICY: Any billing discrepancy under $500 can be resolved
directly by the Billing team without escalation. Disputes over $500, or any
dispute involving a claim of unauthorized charges, must be escalated to
Legal within 24 hours. Refunds for duplicate charges are processed
automatically within 5 business days once confirmed.

SERVICE OUTAGE / TECHNICAL ESCALATION POLICY: Any report of a full service
outage, data loss, or security breach must be classified as High urgency
and routed immediately to Engineering. Partial degradation (e.g. slow
performance, intermittent errors) is Medium urgency unless the customer is
on an Enterprise SLA tier, in which case it is automatically High urgency
regardless of severity, per contractual response-time guarantees.

SALES / UPGRADE POLICY: Requests to add seats, upgrade plan tiers, or
purchase add-ons should be routed to Sales. Sales inquiries are generally
Low to Medium urgency unless the customer explicitly states they will
switch to a competitor without a same-day response, in which case treat
as High urgency retention risk.

LEGAL / COMPLIANCE ESCALATION POLICY: Any mention of legal action, threats
of litigation, regulatory complaints (e.g. GDPR, CCPA), data privacy
violations, or requests for legal documentation (contracts, NDAs, DPAs)
must be routed to Legal and flagged with a risk score of 8 or higher
regardless of the customer's stated emotional tone.

CHURN / RETENTION RISK POLICY: Any explicit statement of intent to cancel,
switch providers, or "this is my last call before I leave" language should
be scored with elevated risk (7+) even if the customer's tone is calm, since
sentiment does not reliably predict churn intent. These calls should be
flagged for the Retention or Billing team with an action item to offer a
retention path (discount, plan downgrade, or account review call) within
24 hours.

FRUSTRATED CUSTOMER DE-ESCALATION POLICY: When a customer expresses strong
frustration or anger, regardless of the underlying issue category, the
recommended action_required should include a callback commitment within
2 business hours to prevent further escalation, in addition to the
department-specific remediation step.
"""


def _split_into_chunks(document: str) -> list:
    """Split the policy document into paragraph-level chunks for retrieval."""
    chunks = [c.strip() for c in document.strip().split("\n\n") if c.strip()]
    return chunks


def _tokenize(text: str) -> list:
    """Lowercase, alphanumeric-only tokenizer used by the fallback vectorizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _FallbackVectorizer:
    """
    Dependency-free bag-of-words vectorizer (term-frequency) with cosine
    similarity, used only if sentence-transformers is unavailable.
    """

    def __init__(self, corpus: list):
        self.vocab = {}
        for doc in corpus:
            for token in _tokenize(doc):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def transform(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for token in _tokenize(text):
            idx = self.vocab.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Standard cosine similarity with a small epsilon to avoid div-by-zero."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class PolicyRAG:
    """
    Retrieval tool over the hardcoded internal policy document.

    Usage:
        rag = PolicyRAG()
        result = rag.search("I want to cancel my enterprise contract")
        # result -> {"matched_chunk": "...", "score": 0.71, "backend": "minilm"}
    """

    def __init__(self):
        self.chunks = _split_into_chunks(INTERNAL_POLICY_DOCUMENT)
        self.backend = None
        self._model = None
        self._chunk_embeddings = None
        self._fallback_vectorizer = None

        # Try to load the real sentence-transformers model. If it's not
        # installed, or fails to download (no network / no HF access),
        # transparently fall back to the TF cosine-similarity method.
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._chunk_embeddings = self._model.encode(self.chunks)
            self.backend = "minilm"
        except Exception:
            self._fallback_vectorizer = _FallbackVectorizer(self.chunks)
            self._chunk_embeddings = np.array(
                [self._fallback_vectorizer.transform(c) for c in self.chunks]
            )
            self.backend = "tf_cosine_fallback"

    def search(self, query: str, top_k: int = 1) -> dict:
        """
        Retrieve the most relevant policy chunk(s) for a given query.

        Args:
            query: the customer transcript or intent summary.
            top_k: number of top chunks to return (default 1).

        Returns:
            dict with:
                matched_chunk: the single best-matching chunk (str)
                score: cosine similarity score of the best match (float)
                backend: which embedding backend was used (str)
                top_matches: list of (chunk, score) tuples, length top_k
        """
        if self.backend == "minilm":
            query_vec = self._model.encode([query])[0]
        else:
            query_vec = self._fallback_vectorizer.transform(query)

        scores = [
            _cosine_similarity(query_vec, chunk_vec)
            for chunk_vec in self._chunk_embeddings
        ]

        ranked = sorted(
            zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        top_matches = ranked[:top_k]

        best_chunk, best_score = top_matches[0] if top_matches else ("", 0.0)

        return {
            "matched_chunk": best_chunk,
            "score": round(best_score, 4),
            "backend": self.backend,
            "top_matches": top_matches,
        }


# Module-level singleton so the (potentially expensive) model load only
# happens once per process, not once per Streamlit rerun.
_rag_singleton = None


def get_rag_engine() -> PolicyRAG:
    """Lazy-load and cache a single PolicyRAG instance for the app session."""
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = PolicyRAG()
    return _rag_singleton
