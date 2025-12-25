from typing import Any, Dict, List, Optional

from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
    PINECONE_HOST,
)

# -------------------------
# Init Pinecone index
# -------------------------
_pc = Pinecone(api_key=PINECONE_API_KEY)

# For Pinecone serverless / newer deployments, passing host is recommended.
if PINECONE_HOST:
    _index = _pc.Index(PINECONE_INDEX, host=PINECONE_HOST)
else:
    _index = _pc.Index(PINECONE_INDEX)


# -------------------------
# Public API
# -------------------------
def upsert(vectors: List[Dict[str, Any]]) -> None:
    """
    vectors example:
      [{"id": "xxx", "values": [..], "metadata": {...}}, ...]
    """
    _index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)


def query(vector: List[float], top_k: int, filter_: Optional[Dict[str, Any]] = None):
    res = _index.query(
        namespace=PINECONE_NAMESPACE,
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_ or None,
    )

    # Pinecone SDK may return a QueryResponse object, not a plain dict.
    if hasattr(res, "to_dict"):
        return res.to_dict()
    if isinstance(res, dict):
        return res
    # last resort
    try:
        return dict(res)
    except Exception:
        return {"matches": []}

