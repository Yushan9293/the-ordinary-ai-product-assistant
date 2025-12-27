from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import unicodedata

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
    PINECONE_NAMESPACE,
)
from pinecone_store import query as pc_query


# -------------------------
# Models
# -------------------------
emb = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model=OPENAI_EMBED_MODEL,
)

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_CHAT_MODEL,
    temperature=0.2,
)


# -------------------------
# System Prompt
# -------------------------
SYSTEM = """
You are a friendly, warm, and knowledgeable skincare consultant for The Ordinary.

IMPORTANT RAG RULES (hard rules):
- You MUST base product names, prices, and URLs ONLY on the retrieved knowledge base / JSON metadata.
- If a specific field (e.g. price, URL, size) is missing in the retrieved metadata:
  - Answer normally using the available information.
  - Explicitly state "not found in the knowledge base" ONLY for the missing field.
- Do NOT invent or guess prices, URLs, or sizes.

Your tone:
- gentle and reassuring
- feminine, friendly, and natural
- like chatting with a real skincare advisor ✨

Guidelines:
- Use light emojis occasionally (💧🌿✨)
- Avoid sounding like a product catalog
- Prefer natural explanations over bullet-heavy lists
- Recommend products in a conversational way
- Do not overwhelm the user with too many options

Answer structure (preferred):
- Short friendly opening
- Product recommendations (numbered, short explanations)
- Simple AM / PM routine
- Safety guidance (patch test, introduce slowly)
- Optional one clarification question at the end (only one)

Always prioritize clarity, safety, and simplicity.
""".strip()

TOP_K = 12


# -------------------------
# Helpers
# -------------------------
def _safe_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _extract_title(meta: Dict[str, Any]) -> Optional[str]:
    # JSON ingestion uses "product_name"
    for k in ("title", "name", "product_name"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_text(meta: Dict[str, Any]) -> str:
    # Prefer enriched text fields
    for k in ("text", "content", "chunk", "description"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_price(meta: Dict[str, Any]) -> Optional[float]:
    # KB often uses price_eur_original
    for k in ("price", "price_eur_original", "price_eur"):
        v = meta.get(k)
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None


def _format_price_eur(p: Optional[float]) -> Optional[str]:
    if p is None:
        return None
    try:
        return f"€{float(p):.2f}"
    except Exception:
        return None


def _extract_url(meta: Dict[str, Any]) -> Optional[str]:
    for k in ("url", "product_url", "link"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _normalize_title_key(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _normalize_matches(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    matches = raw.get("matches", [])
    if not isinstance(matches, list):
        return []

    out: List[Dict[str, Any]] = []
    for m in matches:
        meta = _safe_meta(m.get("metadata"))
        out.append(
            {
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": meta,
                "title": _extract_title(meta) or "Untitled",
                "price": _extract_price(meta),
                "price_display": _format_price_eur(_extract_price(meta)),
                "url": _extract_url(meta),
                "text": _extract_text(meta),
                "source": meta.get("source"),
                "category": meta.get("category"),
                "_type": meta.get("_type") or meta.get("type"),
            }
        )
    return out


def _pick_products(matches: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    """
    Prefer JSON products (they have URL + richer metadata).
    """
    products = []
    for m in matches:
        src = str(m.get("source") or "").lower()
        if src == "json":
            products.append(m)

    products.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return products[:max_items]


def _pick_kb(matches: List[Dict[str, Any]], max_items: int = 3) -> List[Dict[str, Any]]:
    kb = [m for m in matches if str(m.get("source") or "").lower() in ("kb", "md")]
    kb.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return kb[:max_items]


def _format_context(products: List[Dict[str, Any]], kb: List[Dict[str, Any]]) -> str:
    lines: List[str] = []

    if products:
        lines.append("PRODUCT CANDIDATES (from vector search):")
        for i, p in enumerate(products, 1):
            title = p.get("title") or "(untitled)"
            price = p.get("price_display")
            url = p.get("url")
            cat = p.get("category")
            header = f"{i}. {title}"
            if price:
                header += f" | price: {price}"
            if url:
                header += f" | url: {url}"
            if cat:
                header += f" | category: {cat}"
            lines.append(header)
            desc = (p.get("text") or "").strip()
            if desc:
                lines.append(f"   - {desc[:350]}")
            lines.append("")

    if kb:
        lines.append("KNOWLEDGE BASE SNIPPETS:")
        for i, k in enumerate(kb, 1):
            desc = (k.get("text") or "").strip()
            if desc:
                lines.append(f"{i}. {desc[:400]}")
        lines.append("")

    return "\n".join(lines).strip()


def fill_missing_prices_and_urls(products: List[Dict[str, Any]]) -> None:
    """
    If the first retrieval pulled partial metadata, try to enrich by re-querying with the product title.
    We look for the best match with same normalized title key, and copy missing price/url.
    """
    for p in products:
        if p.get("price") is not None and p.get("url"):
            continue

        title = (p.get("title") or "").strip()
        if not title:
            continue

        vec = emb.embed_query(title)
        raw = pc_query(vector=vec, top_k=5)
        matches = _normalize_matches(raw)

        best = None
        tk = _normalize_title_key(title)
        for m in matches:
            if _normalize_title_key(m.get("title") or "") == tk:
                best = m
                break
        if best is None and matches:
            best = matches[0]

        if best:
            if p.get("price") is None and best.get("price") is not None:
                p["price"] = best["price"]
                p["price_display"] = _format_price_eur(best["price"])
            if not p.get("url") and best.get("url"):
                p["url"] = best["url"]


def _canon_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()
    
    # 去掉不可见空白（含 NBSP 等）
    s = re.sub(r"\s+", " ", s)
    return s

def merge_products_keep_best(products: list[dict]) -> list[dict]:
    best = {}
    for p in products:
        name = p.get("title") or p.get("name") or ""
        k = _canon_key(name)
        if not k:
            continue

        if k not in best:
            best[k] = p
            continue

        cur = best[k]

        # “字段更全”的优先：谁有值用谁
        for field in ["url", "price", "price_display", "category", "source"]:
            if (cur.get(field) in [None, "", "N/A"]) and (p.get(field) not in [None, "", "N/A"]):
                cur[field] = p.get(field)

        # 如果你有 score，也可以让 score 更高的优先
        if (cur.get("score") is None) and (p.get("score") is not None):
            cur["score"] = p["score"]

        best[k] = cur

    return list(best.values())

def _is_complete(p: dict) -> bool:
    has_url = bool(p.get("url"))
    has_price = bool(p.get("price") or p.get("price_display"))
    return has_url and has_price



# -------------------------
# Main
# -------------------------
def answer(
    question: str,
    category: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    # 1) embed question -> query pinecone
    vec = emb.embed_query(question)
    raw = pc_query(vector=vec, top_k=TOP_K)
    matches = _normalize_matches(raw)

    # 2) pick products + kb
    products = _pick_products(matches, max_items=6)
    kb = _pick_kb(matches, max_items=3)

    # 3) enrich (price/url)
    fill_missing_prices_and_urls(products)
    products = merge_products_keep_best(products)
    products_complete = [p for p in products if _is_complete(p)]
    final_recos = (products_complete + products)[:3]

    # 4) build context -> LLM
    ctx = _format_context(final_recos, kb)


    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    if history:
        # keep short history, but do not rely on it for prices/urls
        messages.extend(history[-8:])
    messages.append({"role": "user", "content": f"Context:\n{ctx}\n\nUser question:\n{question}"})

    resp = llm.invoke(messages)
    answer_text = getattr(resp, "content", str(resp))

    # 5) structured output for API (buy/price shortcut memory)
    recommended_products: List[Dict[str, Any]] = []
    for p in final_recos:
        name = p.get("title") or p.get("name") or ""
        recommended_products.append(
            {
                "name": name,
                "title": name,
                "price": p.get("price"),
                "price_display": p.get("price_display"),
                "url": p.get("url"),
                "category": p.get("category"),
                "source": p.get("source"),
            }
        )

    return {
        "answer": answer_text,
        "retrieved_total": len(matches),
        "retrieved_products": len(products),
        "namespace": PINECONE_NAMESPACE,
        "category": category,
        "recommended_products": recommended_products,
        "version": "2025-12-24-rag-v2",
    }
