from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI
from pydantic import BaseModel

from config import PINECONE_NAMESPACE, VERSION
from rag import answer as rag_answer


app = FastAPI()


# -------------------------
# In-memory conversation + state (demo)
# -------------------------
_HISTORY: Dict[str, List[Dict[str, str]]] = {}
_STATE: Dict[str, Dict[str, Any]] = {}


def _get_state(user_id: str) -> Dict[str, Any]:
    s = _STATE.get(user_id)
    if not s:
        s = {
            "last_recommended_full": [],  # [{name,title,price,price_display,url,category,source,score}]
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
    user_id: str = "test"
    category: Optional[str] = None


# -------------------------
# Simple intent helpers
# -------------------------
def detect_buy_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["buy", "purchase", "order", "i want to buy", "i'll take", "send me the link", "second one",
                                "acheter", "je veux", "je voudrais", "commander",
                                "买", "购买", "我要买", "我想买", "下单", "链接", "第二个", "第二"])


def detect_price_followup(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["price", "how much", "cost", "exact price", "prix", "combien", "多少钱", "价格", "价钱"])



def pick_product_choice(text: str, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Very small heuristic: if user says "second", pick index 1, etc.
    Otherwise None.
    """
    t = text.lower().strip()
    if not options:
        return None

    if "first" in t or t == "1":
        return options[0] if len(options) >= 1 else None
    if "second" in t or t == "2":
        return options[1] if len(options) >= 2 else None
    if "third" in t or t == "3":
        return options[2] if len(options) >= 3 else None

    return None


def build_price_followup_answer(options: List[Dict[str, Any]]) -> str:
    if not options:
        return "I don't have any recently recommended products to price-check. Please ask for a recommendation first."
    lines = []
    for i, p in enumerate(options[:3], start=1):
        name = p.get("title") or p.get("name") or f"Option {i}"
        price_display = p.get("price_display") or (f"€{p.get('price'):.2f}" if isinstance(p.get("price"), (int, float)) else None)
        if price_display:
            lines.append(f"{i}. {name}: {price_display}")
        else:
            lines.append(f"{i}. {name}: price not found")
    return "Here are the prices I have from the retrieved data:\n" + "\n".join(lines)


# -------------------------
# API
# -------------------------
@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    user_id = req.user_id
    question = req.question

    # ✅ must initialize state/history first (used by shortcuts)
    state = _get_state(user_id)
    history = get_history(user_id)
    
    # ✅ 0) CHOICE shortcut: user replies with 1/2/3 to pick from last recommendations
    if question.strip() in {"1", "2", "3"} and state.get("last_recommended_full"):
        idx = int(question.strip()) - 1
        options = state["last_recommended_full"]
        if 0 <= idx < len(options):
            chosen = options[idx]
            name = chosen.get("name") or chosen.get("title") or "this product"
            url = chosen.get("url")
            price = chosen.get("price_display") or (f"€{chosen.get('price'):.2f}" if isinstance(chosen.get("price"), (int, float)) else None)
            price_text = price or "price not found"

            if url:
                answer_text = f"URL: {url}\nPrice: {price_text}"
            else:
                answer_text = f"I found **{name}**, but the official URL is not available in the catalog."

            save_message(user_id, "user", question)
            save_message(user_id, "assistant", answer_text)
            return {
                "answer": answer_text,
                "purchase": {"name": name, "url": url, "price": chosen.get("price"), "price_display": chosen.get("price_display")},
                "recommended_products": state["last_recommended_full"],
                "retrieved_total": 0,
                "retrieved_products": 0,
                "namespace": PINECONE_NAMESPACE,
                "version": VERSION,
            }


    # ✅ 1) PRICE FOLLOW-UP shortcut (MUST be before buy + before calling RAG)
    if detect_price_followup(question) and state.get("last_recommended_full") and not detect_buy_intent(question):
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

        if chosen:
            url = chosen.get("url")
            name = chosen.get("name") or chosen.get("title") or "this product"

            price = chosen.get("price_display") or (f"€{chosen.get('price'):.2f}" if isinstance(chosen.get("price"), (int, float)) else None)
            price_text = price or "price not found"

            if url:
                answer_text = f"URL: {url}\nPrice: {price_text}"
            else:
                answer_text = f"I found **{name}**, but the official URL is not available in the catalog."


            save_message(user_id, "user", question)
            save_message(user_id, "assistant", answer_text)

            return {
                "answer": answer_text,
                "purchase": {"name": name, "url": url, "price": chosen.get("price"), "price_display": chosen.get("price_display")},
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
            "Please reply with 1, 2, or 3 (or type the product name).\n\n"
            f"Options: {options}"
        )

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

    # 3) normal: call RAG
    result = rag_answer(question, category=req.category, history=history)

    answer_text = str(result.get("answer", "")) if isinstance(result, dict) else str(result)
    final_recos = result.get("recommended_products", []) if isinstance(result, dict) else []

    # ✅ store last recommended for price/buy shortcuts
    state["last_recommended_full"] = final_recos
    state["updated_at"] = time.time()

    save_message(user_id, "user", question)
    save_message(user_id, "assistant", answer_text)

    return {
        "answer": answer_text,
        "recommended_products": final_recos,
        "retrieved_total": result.get("retrieved_total", 0) if isinstance(result, dict) else 0,
        "retrieved_products": result.get("retrieved_products", 0) if isinstance(result, dict) else 0,
        "namespace": result.get("namespace", PINECONE_NAMESPACE) if isinstance(result, dict) else PINECONE_NAMESPACE,
        "version": result.get("version", VERSION) if isinstance(result, dict) else VERSION,
    }
