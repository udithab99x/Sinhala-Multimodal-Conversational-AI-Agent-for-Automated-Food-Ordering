"""
RAG — Retrieval-Augmented Generation for restaurant menu lookup.
Uses sentence-transformers embeddings + FAISS vector index.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, good multilingual quality


class MenuRAG:
    """
    Stores menu items as dense vectors; retrieves relevant items
    given a natural-language customer query.
    """

    def __init__(self, menu_path: Path):
        with open(menu_path, "r", encoding="utf-8") as f:
            self._menu = json.load(f)

        self._encoder = SentenceTransformer(EMBED_MODEL)
        self._items = self._menu["items"]
        self._offers = self._menu.get("special_offers", [])
        self._restaurant = self._menu["restaurant"]

        # Build corpus: one string per item combining all searchable fields
        self._corpus = [self._item_to_text(item) for item in self._items]
        self._corpus += [self._offer_to_text(o) for o in self._offers]
        self._all_docs = self._items + self._offers

        # Encode and index
        embeddings = self._encoder.encode(self._corpus, convert_to_numpy=True)
        self._index = faiss.IndexFlatL2(embeddings.shape[1])
        self._index.add(embeddings.astype(np.float32))

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top-k most relevant menu items for the query."""
        vec = self._encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self._index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._all_docs):
                doc = dict(self._all_docs[idx])
                doc["_score"] = float(dist)
                results.append(doc)
        return results

    def get_item_by_name(self, name: str) -> dict | None:
        """Exact or fuzzy name lookup."""
        name_l = name.lower()
        for item in self._items:
            if name_l in item["name"].lower() or name_l in item["sinhala"]:
                return item
        return None

    def get_menu_summary(self) -> str:
        """Return a compact text summary of the menu for the system prompt."""
        categories: dict[str, list] = {}
        for item in self._items:
            cat = item["category"]
            categories.setdefault(cat, []).append(item)

        lines = [f"Shop: {self._restaurant['name']} — Pickup only", ""]
        for cat, items in categories.items():
            lines.append(f"[{cat.upper()}]")
            for it in items:
                avail = "✓" if it.get("available", True) else "✗ unavailable"
                lines.append(f"  {it['name']} ({it['sinhala']}) {avail}")
                lines.append(f"    {self._price_text(it)}")
            lines.append("")
        return "\n".join(lines)

    def get_context_for_query(self, query: str) -> str:
        """Retrieve relevant menu items and format them as context text."""
        hits = self.search(query, top_k=4)
        if not hits:
            return "No relevant menu items found."
        lines = ["Relevant menu items:"]
        for h in hits:
            avail = "Available" if h.get("available", True) else "NOT available"
            lines.append(f"- {h['name']} ({h.get('sinhala','')}) — {avail}")
            lines.append(f"  {self._price_text(h)}")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _price_text(self, item: dict) -> str:
        """Format pricing depending on item type (options/packages/portions)."""
        if "options" in item:
            parts = [f"{o['protein']} Rs.{o['price_lkr']}" for o in item["options"]]
            return "Protein options: " + " | ".join(parts)
        if "packages" in item:
            parts = [f"{p['count']}pcs Rs.{p['price_lkr']}" for p in item["packages"]]
            return "Packages: " + " | ".join(parts)
        if "portions" in item:
            parts = [f"{p['size']} Rs.{p['price_lkr']}" for p in item["portions"]]
            return "Portions: " + " | ".join(parts)
        if "price_lkr" in item:
            return f"Rs.{item['price_lkr']}"
        return ""

    def _item_to_text(self, item: dict) -> str:
        parts = [item["name"], item.get("sinhala", ""), item.get("category", ""),
                 item.get("description", ""), self._price_text(item)]
        return " ".join(str(p) for p in parts if p)

    def _offer_to_text(self, offer: dict) -> str:
        return f"{offer['name']} {offer.get('sinhala','')} {offer.get('description','')}"
