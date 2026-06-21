"""
Core conversation agent — Wakwalle Kade food shop ordering system.
Uses the fine-tuned Gemma 4 E2B model + RAG menu retrieval.
Maintains per-user session state. Saves confirmed orders to CSV.
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from app.rag import MenuRAG
from app.config import settings


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """ඔබ Wakwalle Kade කෑම කඩේ AI food ordering assistant කෙනෙකි.
ඔබ Sinhala, English, සහ mixed Sinhala-English භාෂාවෙන් customers සමඟ කතා කරනවා.

මෙනු:
Rice & Curry (protein choose කරන්න):
  - Veg (එළවළු): Rs.250
  - Egg (බිත්තර): Rs.290
  - Omelette (ඔම්ලට්): Rs.320
  - Fish (මාළු): Rs.350
  - Chicken (චිකන්): Rs.400

String Hoppers (ඉදිආප්ප) — package choose කරන්න:
  - 10pcs: Rs.150  |  15pcs: Rs.225  |  20pcs: Rs.300  |  30pcs: Rs.450

Kottu — half/full portion:
  - Chicken Kottu: Half Rs.450 / Full Rs.800
  - Egg Kottu: Half Rs.350 / Full Rs.600
  - Fish Kottu: Half Rs.420 / Full Rs.750

Fried Rice — half/full portion:
  - Chicken Fried Rice: Half Rs.450 / Full Rs.800
  - Egg Fried Rice: Half Rs.350 / Full Rs.600
  - Fish Fried Rice: Half Rs.420 / Full Rs.750

ඔබේ ordering flow:
1. Customer item identify කරන්න
2. Rice & Curry නම් → protein (veg/egg/omelette/fish/chicken) අසන්න
3. Kottu / Fried Rice නම් → portion size (half/full) අසන්න
4. String Hoppers නම් → package (10/15/20/30) අසන්න
5. Quantity confirm කරන්න
6. Phone number collect කරන්න
7. Full order summary with total show කරන්න
8. Confirm/cancel ask කරන්න

Rules:
- No delivery — pickup only (කඩේ ලඟට ගන්නෝ ගන්න ඕනේ)
- NEVER ask about spice levels
- Always collect phone number before confirming
- Show order summary with total before final confirmation
- Be friendly, short, and concise
- Respond in the same language the customer uses (Sinhala / English / mixed)

{menu_context}"""


# ── Order & session state ─────────────────────────────────────────────────────

@dataclass
class OrderItem:
    name: str
    portion: str | None = None      # "half"|"full"|"10pcs"|"15pcs"|"veg"|"chicken" etc.
    quantity: int = 1
    price: int = 0


@dataclass
class Session:
    user_id: str
    history: list[dict] = field(default_factory=list)
    items: list[OrderItem] = field(default_factory=list)
    phone_number: str | None = None
    confirmed: bool = False
    order_id: str | None = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self):
        self.last_active = time.time()

    def is_expired(self, ttl_seconds: int = 1800) -> bool:
        return time.time() - self.last_active > ttl_seconds

    def order_total(self) -> int:
        return sum(i.price * i.quantity for i in self.items)

    def order_summary(self) -> str:
        if not self.items:
            return "Order is empty."
        lines = ["Order Summary / ඇණවුම:"]
        for it in self.items:
            portion_str = f" ({it.portion})" if it.portion else ""
            lines.append(
                f"  {it.name}{portion_str} x{it.quantity} — Rs.{it.price * it.quantity}"
            )
        lines.append(f"  ──────────────────────")
        lines.append(f"  Total — Rs.{self.order_total()}")
        lines.append(f"  Pickup only / කඩේ ලඟට ගන්න")
        if self.phone_number:
            lines.append(f"  Phone: {self.phone_number}")
        return "\n".join(lines)


# ── CSV order logger ──────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent.parent / "data" / "orders.csv"
CSV_HEADERS = ["order_id", "timestamp", "phone_number", "channel", "items", "total_lkr"]


def _ensure_csv():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def save_order_to_csv(session: Session, channel: str = "whatsapp") -> str:
    """Save a confirmed order to the CSV file. Returns the order ID."""
    _ensure_csv()
    order_id = f"WK{int(time.time()) % 100000:05d}"
    items_str = " | ".join(
        f"{it.name} ({it.portion or 'full'}) x{it.quantity} Rs.{it.price * it.quantity}"
        for it in session.items
    )
    row = {
        "order_id": order_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phone_number": session.phone_number or "unknown",
        "channel": channel,
        "items": items_str,
        "total_lkr": session.order_total(),
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)
    return order_id


# ── LLM backend ───────────────────────────────────────────────────────────────

class LLMBackend:
    """Loads the fine-tuned Gemma 4 E2B model from HuggingFace."""

    def __init__(self, model_id: str, hf_token: str | None = None):
        print(f"[LLM] Loading: {model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token or None,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            return_full_text=False,
        )
        print("[LLM] Ready.")

    def generate(self, messages: list[dict]) -> str:
        output = self._pipe(messages)
        return output[0]["generated_text"].strip()


# ── Main agent ────────────────────────────────────────────────────────────────

class FoodOrderingAgent:
    """Stateful conversational agent keyed by user_id (WhatsApp/voice number)."""

    def __init__(self, menu_path: Path, model_id: str, hf_token: str = ""):
        self._rag = MenuRAG(menu_path)
        self._llm = LLMBackend(model_id, hf_token)
        self._sessions: dict[str, Session] = {}
        _ensure_csv()

    def respond(self, user_id: str, user_message: str, channel: str = "whatsapp") -> str:
        session = self._get_or_create_session(user_id)
        session.touch()

        rag_context = self._rag.get_context_for_query(user_message)
        system = SYSTEM_PROMPT.format(menu_context=rag_context)

        messages = [{"role": "system", "content": system}]
        messages += session.history[-10:]
        messages.append({"role": "user", "content": user_message})

        reply = self._llm.generate(messages)

        session.history.append({"role": "user", "content": user_message})
        session.history.append({"role": "assistant", "content": reply})

        self._update_state(session, user_message, channel)

        return reply

    def reset_session(self, user_id: str):
        self._sessions.pop(user_id, None)

    def get_session(self, user_id: str) -> Session | None:
        return self._sessions.get(user_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_or_create_session(self, user_id: str) -> Session:
        s = self._sessions.get(user_id)
        if s is None or s.is_expired():
            s = Session(user_id=user_id)
            self._sessions[user_id] = s
        return s

    def _update_state(self, session: Session, user_msg: str, channel: str):
        # Extract phone number (Sri Lankan format)
        phone_m = re.search(
            r"(\+94|0)?[0-9]{9,10}|07[0-9]{8}", user_msg
        )
        if phone_m and not session.phone_number:
            session.phone_number = phone_m.group(0).strip()

        # Check confirmation
        if re.search(r"\b(yes|confirm|ඔව්|yep|හරි|ස්ථිර)\b", user_msg, re.IGNORECASE):
            if session.phone_number and session.items:
                if not session.confirmed:
                    session.confirmed = True
                    session.order_id = save_order_to_csv(session, channel)

        # Check cancellation
        if re.search(r"\b(cancel|no|nah|bahe|එපා)\b", user_msg, re.IGNORECASE):
            session.confirmed = False


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent: FoodOrderingAgent | None = None


def get_agent() -> FoodOrderingAgent:
    global _agent
    if _agent is None:
        _agent = FoodOrderingAgent(
            menu_path=settings.menu_path,
            model_id=settings.hf_model_id,
            hf_token=settings.hf_token,
        )
    return _agent
