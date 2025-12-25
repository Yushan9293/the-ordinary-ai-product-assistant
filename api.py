from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import time

from fastapi import FastAPI
from pydantic import BaseModel

from config import PINECONE_NAMESPACE
from rag import answer as rag_answer

app = FastAPI(title="The Ordinary RAG API")

VERSION = "2025-12-24-api-v2"

# -------------------------
# In-memory conversation + state (demo)
# -------------------------
_HISTORY: Dict[str, List[Dict[str, str]]] = {}
_STATE: Dict[str, Dict[str, Any]] = {}


def _get_state(user_id: str) -> Dict[str, Any]:
    s = _STATE.get(user_id)
    if not s:
        s = {
            "last_recommended_full": [],  # [{name,title,price,price_display,url,category,source}, ...]
            "updated_at": time.time(),
        }
        _STATE[user_id] = s
    return s


def save_message(user_id: str, role: str, content: str) -> None:
    _HISTORY.setdefault(user_id, []).append({"role": role, "content": content})


def get_history(user_id: str, max_turns: int = 12) -> List[Dict[str, str]]:
    hist = _HISTORY.get(user_id, [])
    return hist[-max_turns:]


# -------------------------
# Request model
# -------------------------
class QueryRequest(BaseModel):
    question: str
    user_id: str
    category: Optional[str] = None


# -------------------------
# Intent helpers
# -------------------------
_BUY_RE = re.compile(r"\b(buy|purchase|order|checkout|get it|i want it)\b", re.I)
_PRICE_RE = re.compile(r"\b(price|how much|exactly|cost|prix|combien|多少钱|价格)\b", re.I)


def detect_buy_intent(text: str) -> bool:
    return bool(_BUY_RE.search(text or ""))


def detect_price_followup(text: str) -> bool:
    return bool(_PRICE_RE.search(text or ""))


def pick_product_choice(user_text: str, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    t = (user_text or "").strip().lower()

    # numeric choice: "2"
    m = re.search(r"\b([1-9])\b", t)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(items):
            return items[idx]

    # "second one"
    order_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    for k, i in order_map.items():
        if k in t and 0 <= i < len(items):
            return items[i]

    # match by name substring
    for it in items:
        name = (it.get("name") or it.get("title") or "").lower()
        if name and name in t:
            return it

    return None


def build_price_followup_answer(items: List[Dict[str, Any]]) -> str:
    lines = ["Here are the exact prices from the knowledge base:"]
    for i, it in enumerate(items, 1):
        name = it.get("name") or it.get("title") or f"Item {i}"
        price_display = it.get("price_display")
        if not price_display:
            lines.append(f"{i}. {name} — price not found in the knowledge base")
        else:
            lines.append(f"{i}. {name} — {price_display}")
    return "\n".join(lines)


# -------------------------
# Main endpoint
# -------------------------
@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    user_id = req.user_id
    question = req.question

    state = _get_state(user_id)

    # ✅ 1) PRICE FOLLOW-UP shortcut (MUST be before buy + before calling RAG)
    if detect_price_followup(question) and state.get("last_recommended_full"):
        answer_text = build_price_followup_answer(state["last_recommended_full"])

        save_message(user_id, "user", question)
        save_message(user_id, "assistant", answer_text)

        return {
            "answer": answer_text,
            "recommended_products": state["last_recommended_full"],
            "retrieved_total": 0,
            "retrieved_products": 0,
            "namespace": PINECONE_NAMESPACE,
            "version": VERSION,
        }

    # ✅ 2) BUY shortcut (use memory, do NOT call RAG)
    if detect_buy_intent(question) and state.get("last_recommended_full"):
        chosen = pick_product_choice(question, state["last_recommended_full"])

        # clear choice
        if chosen:
            url = chosen.get("url")
            name = chosen.get("name") or chosen.get("title") or "this product"

            if url:
                answer_text = (
                    f"Perfect! Here is the official link to buy **{name}**:\n{url}\n\n"
                    "You can click the link to complete your purchase."
                )
            else:
                answer_text = f"I found **{name}**, but the URL is not found in the knowledge base."

            save_message(user_id, "user", question)
            save_message(user_id, "assistant", answer_text)

            return {
                "answer": answer_text,
                "purchase": {"name": name, "url": url},
                "recommended_products": state["last_recommended_full"],
                "retrieved_total": 0,
                "retrieved_products": 0,
                "namespace": PINECONE_NAMESPACE,
                "version": VERSION,
            }

        # unclear choice -> ask user to pick
        options = [
            {"index": i + 1, "name": (p.get("name") or p.get("title")), "url": p.get("url")}
            for i, p in enumerate(state["last_recommended_full"])
        ]
        answer_text = (
            "Sure — which one would you like to buy?\n"
            "Please reply with 1, 2, or 3 (or type the product name)."
        )

        save_message(user_id, "user", question)
        save_message(user_id, "assistant", answer_text)

        return {
            "answer": answer_text,
            "options": options,
            "retrieved_total": 0,
            "retrieved_products": 0,
            "namespace": PINECONE_NAMESPACE,
            "version": VERSION,
        }

    # 3) Normal RAG flow
    history = get_history(user_id)
    result = rag_answer(question=question, category=req.category, history=history)

    save_message(user_id, "user", question)
    save_message(user_id, "assistant", str(result.get("answer", "")))

    # ✅ store last recommended for price/buy shortcuts
    if isinstance(result, dict) and isinstance(result.get("recommended_products"), list):
        state["last_recommended_full"] = result["recommended_products"][:3]
        state["updated_at"] = time.time()

    if isinstance(result, dict):
        result["version"] = VERSION
        return result

    return {"answer": str(result), "version": VERSION}
