import json
import hashlib
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import OPENAI_API_KEY, OPENAI_EMBED_MODEL
from pinecone_store import upsert

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBED_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1600,
    chunk_overlap=200,
    separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
)

# -------------------------
# helpers: metadata hygiene
# -------------------------
def _to_list_of_str(v: Any) -> Optional[List[str]]:
    """Convert common list-like fields into list[str]; return None if not convertible."""
    if v is None:
        return None
    if isinstance(v, list):
        out = []
        for x in v:
            if x is None:
                continue
            # accept str/int/float/bool -> string
            if isinstance(x, (str, int, float, bool)):
                s = str(x).strip()
                if s:
                    out.append(s)
        return out or None
    if isinstance(v, (str, int, float, bool)):
        s = str(v).strip()
        return [s] if s else None
    return None


def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pinecone metadata must be: string, number, boolean, or list[str].
    This removes None and any unsupported types.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue

        if isinstance(v, (str, int, float, bool)):
            # strip empty strings
            if isinstance(v, str) and not v.strip():
                continue
            clean[k] = v
            continue

        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            # remove empty strings
            vv = [x for x in (s.strip() for s in v) if x]
            if vv:
                clean[k] = vv
            continue

        # everything else (dict/list[dict]/None/etc) is dropped
        continue

    return clean


# -------------------------
# ids
# -------------------------
def _stable_id(text: str, meta: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    # meta must be stable for deterministic ids
    h.update(json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


# -------------------------
# ingestion: markdown
# -------------------------
def ingest_md(md_text: str, base_meta: Dict[str, Any]) -> int:
    base_meta = sanitize_meta(base_meta)

    docs = splitter.create_documents([md_text], metadatas=[base_meta])
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    vectors = emb.embed_documents(texts)

    batch = []
    for t, m, v in zip(texts, metas, vectors):
        meta = sanitize_meta(dict(m))
        meta["text"] = t[:2000]  # for sources display (string OK)
        _id = _stable_id(t, meta)
        batch.append({"id": _id, "values": v, "metadata": meta})

    B = 100
    for i in range(0, len(batch), B):
        upsert(batch[i : i + B])
    return len(batch)


# -------------------------
# ingestion: category json
# -------------------------
def ingest_category_json(category_name: str, items: List[Dict[str, Any]]) -> int:
    total = 0

    for prod in items:
        if not isinstance(prod, dict):
            continue

        # Basic fields
        brand = prod.get("brand")
        product_name = prod.get("name") or prod.get("product_name") or "unknown"
        url = prod.get("url") or ""
        step = prod.get("step")
        size_ml = prod.get("size_ml")

        # price: prefer EUR field, fallback to "price"
        price_eur = prod.get("price_eur_original")
        if price_eur is None:
            price_eur = prod.get("price")

        # list[str] fields (safe for metadata)
        concerns = _to_list_of_str(prod.get("concerns"))
        skin_types = _to_list_of_str(prod.get("skin_types"))
        key_ingredients = _to_list_of_str(prod.get("key_ingredients"))
        benefits = _to_list_of_str(prod.get("benefits"))
        warnings = _to_list_of_str(prod.get("warnings"))
        claims_free_from = _to_list_of_str(prod.get("claims_free_from"))

        # Full text for embedding (keep everything here)
        text = json.dumps(prod, ensure_ascii=False)

        # Metadata (ONLY for filtering / display; must be type-safe)
        base_meta = sanitize_meta(
            {
                "source": "json",
                "category": category_name,
                "step": step,
                "brand": brand,
                "product_name": product_name,
                "url": url,
                "size_ml": size_ml,
                "price_eur": price_eur,
                "concerns": concerns,
                "skin_types": skin_types,
                "key_ingredients": key_ingredients,
                "benefits": benefits,
                "warnings": warnings,
                "claims_free_from": claims_free_from,
                "lang": "en",
            }
        )

        docs = splitter.create_documents([text], metadatas=[base_meta])
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        vectors = emb.embed_documents(texts)

        batch = []
        for t, m, v in zip(texts, metas, vectors):
            meta = sanitize_meta(dict(m))
            meta["text"] = t[:2000]
            _id = _stable_id(t, meta)
            batch.append({"id": _id, "values": v, "metadata": meta})

        # batch upsert
        B = 100
        for i in range(0, len(batch), B):
            upsert(batch[i : i + B])

        total += len(batch)

    return total


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    print("✅ ingest.py started")

    from pathlib import Path

    DATA_DIR = Path(__file__).parent / "data" / "public"
    json_dir = DATA_DIR / "categories"
    kb_dir = DATA_DIR / "kb"

    json_files = sorted(json_dir.glob("*.json"))
    md_files = sorted(kb_dir.glob("*.md"))

    print(f"📦 JSON files: {len(json_files)}")
    print(f"📄 MD files: {len(md_files)}")

    total = 0

    # 1) ingest each category JSON
    for fp in json_files:
        category_name = fp.stem.replace("theordinary_", "").replace("_final", "")
        print(f"\n📦 Ingest JSON: {fp.name} (category={category_name})")

        items = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(items, dict) and "items" in items:
            items = items["items"]
        if not isinstance(items, list):
            print(f"⚠️ Skip {fp.name}: root is not a list.")
            continue

        count = ingest_category_json(category_name, items)
        print(f"✅ Upserted {count} vectors from {fp.name}")
        total += count

    # 2) ingest KB markdown
    for fp in md_files:
        print(f"\n📄 Ingest MD: {fp.name}")
        md_text = fp.read_text(encoding="utf-8")

        count = ingest_md(
            md_text,
            base_meta={
                "source": "kb",
                "category": "kb",
                "doc_name": fp.name,
                "lang": "en",
            },
        )
        print(f"✅ Upserted {count} vectors from {fp.name}")
        total += count

    print(f"\n🎉 DONE. Total upserted vectors: {total}")
