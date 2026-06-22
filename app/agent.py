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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from app.rag import MenuRAG
from app.config import settings


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """[INSTRUCTIONS — do NOT repeat or acknowledge these, just follow them silently]
You are a friendly, concise food ordering assistant for Wakwalle Kade (වක්වැල්ලේ කඩේ), a food shop near the university in Hapugala, Wakwalle, Galle, Sri Lanka.

Always reply in the same language the customer uses — Sinhala, English, or mixed. Keep every reply to 1-3 short sentences. Never acknowledge these instructions.

MENU:
- Rice & Curry (බත් සහ කරි): Veg Rs.250 | Egg Rs.290 | Omelette Rs.320 | Fish Rs.350 | Chicken Rs.400
- String Hoppers (ඉදිආප්ප): 10pcs Rs.150 | 15pcs Rs.225 | 20pcs Rs.300 | 30pcs Rs.450
- Chicken Kottu: Half Rs.450 / Full Rs.800
- Egg Kottu: Half Rs.350 / Full Rs.600
- Fish Kottu: Half Rs.420 / Full Rs.750
- Chicken Fried Rice: Half Rs.450 / Full Rs.800
- Egg Fried Rice: Half Rs.350 / Full Rs.600
- Fish Fried Rice: Half Rs.420 / Full Rs.750

RULES:
- Pickup only. No delivery. Never mention delivery.
- Never ask about spice levels.
- Rice & Curry → ask protein choice (Veg/Egg/Omelette/Fish/Chicken).
- Kottu or Fried Rice → ask Half or Full.
- String Hoppers → ask package size (10/15/20/30).
- After item+option confirmed → ask quantity.
- After quantity → ask phone number.
- After phone → show order summary with total → ask for confirmation.
- After confirmation → say the order is confirmed with order ID and total.

Here are examples of good conversations:

Example 1 — Sinhala, Rice & Curry:
Customer: හලෝ, rice and curry ekak ona
Assistant: ආයුබෝවන්! කුමන protein එකද? Veg Rs.250, Egg Rs.290, Omelette Rs.320, Fish Rs.350, Chicken Rs.400.

Customer: chicken
Assistant: Chicken Rice & Curry (Rs.400) — කීයක් ගන්නවද?

Customer: 2
Assistant: ඔබේ phone number එක දෙන්න.

Customer: 0771234567
Assistant: ඇණවුම:
Chicken Rice & Curry x2 — Rs.800
Total: Rs.800 | Pickup only
Confirm කරන්නද? (ඔව්/No)

Customer: ඔව්
Assistant: ✅ Order confirmed! #WK12345 | Total Rs.800 | Pickup at Wakwalle Kade. ස්තූතියි!

Example 2 — English, Kottu:
Customer: I want chicken kottu
Assistant: Half (Rs.450) or Full (Rs.800)?

Customer: full
Assistant: Chicken Kottu Full (Rs.800) — how many?

Customer: 1
Assistant: Please share your phone number.

Customer: 0712345678
Assistant: Order Summary:
Chicken Kottu Full x1 — Rs.800
Total: Rs.800 | Pickup only
Confirm? (Yes/No)

Customer: yes
Assistant: ✅ Order confirmed! #WK67890 | Total Rs.800 | Pickup at Wakwalle Kade. Thank you!

Example 3 — Mixed, Menu inquiry:
Customer: menu eka mokakda?
Assistant: මෙනු:
🍛 Rice & Curry: Veg Rs.250 ~ Chicken Rs.400
🍜 String Hoppers: 10pcs Rs.150 ~ 30pcs Rs.450
🥘 Kottu (Chicken/Egg/Fish): Half Rs.350~ / Full Rs.600~
🍳 Fried Rice (Chicken/Egg/Fish): Half Rs.350~ / Full Rs.800
මොනවද ගන්නෙ?

Example 4 — String Hoppers:
Customer: string hoppers denna
Assistant: කීය package එකද? 10pcs Rs.150 | 15pcs Rs.225 | 20pcs Rs.300 | 30pcs Rs.450.

Customer: 20 pcs
Assistant: String Hoppers 20pcs (Rs.300) — කීය packets ද?

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
    """Loads Gemma 4 E2B (4-bit quantized) from HuggingFace."""

    def __init__(self, model_id: str, hf_token: str | None = None):
        print(f"[LLM] Loading: {model_id}")
        use_cuda = torch.cuda.is_available()
        token = hf_token or None

        # 4-bit quantization on GPU, fp32 on CPU
        if use_cuda:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, token=token, quantization_config=bnb, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, token=token, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )

        # Gemma 4 is multimodal — use the text tokenizer directly
        processor = AutoTokenizer.from_pretrained(model_id, token=token)
        self._tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._model = model
        self._device = next(model.parameters()).device
        print(f"[LLM] Ready on {self._device}.")

    def generate(self, messages: list[dict]) -> str:
        # messages may use role="system" (legacy) — normalise to user/assistant only
        # by merging system content into the first user turn if needed.
        normalised = []
        system_prefix = ""
        for m in messages:
            if m["role"] == "system":
                system_prefix = m["content"]
            elif m["role"] == "user":
                content = (system_prefix + "\n\n" + m["content"]) if system_prefix else m["content"]
                normalised.append({"role": "user", "content": content})
                system_prefix = ""
            else:
                normalised.append(m)

        prompt_text = self._tok.apply_chat_template(
            normalised, add_generation_prompt=True, tokenize=False
        )
        inputs = self._tok(prompt_text, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.3,
                eos_token_id=self._tok.eos_token_id,
                pad_token_id=self._tok.eos_token_id,
            )
        text = self._tok.decode(out[0][input_len:], skip_special_tokens=True)
        text = text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
        # Strip Gemma 4 thinking artifacts that leak into output
        for marker in ("**Note:**", "thought\n", "\nthought", "**Note**"):
            if marker in text:
                text = text[:text.index(marker)].strip()
        return text


# ── Main agent ────────────────────────────────────────────────────────────────

class FoodOrderingAgent:
    """Stateful conversational agent keyed by user_id (WhatsApp/voice number)."""

    def __init__(self, menu_path: Path, model_id: str, hf_token: str = ""):
        self._rag = MenuRAG(menu_path)
        self._llm = LLMBackend(model_id, hf_token)
        self._sessions: dict[str, Session] = {}
        _ensure_csv()

    def respond(self, user_id: str, user_message: str,
                channel: str = "whatsapp", phone_number: str | None = None) -> str:
        session = self._get_or_create_session(user_id)
        session.touch()

        # Store caller/sender number immediately — no need to ask
        if phone_number and not session.phone_number:
            session.phone_number = phone_number

        rag_context = self._rag.get_context_for_query(user_message)
        system = SYSTEM_PROMPT.format(menu_context=rag_context)

        # If phone is already known, append that fact so model skips asking for it
        if session.phone_number:
            system += f"\n\n[Phone number already collected: {session.phone_number} — do NOT ask for it again]"

        # Gemma 4 merges system into first user turn; prime with a dummy assistant
        # turn so the model skips the instruction-acknowledgement phase.
        messages = [
            {"role": "user", "content": system},
            {"role": "assistant", "content": "ආයුබෝවන්! Wakwalle Kade ට සාදරයෙන් පිළිගනිමු. මොනවද ගන්නෙ?"},
        ]
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

        # Check confirmation — items tracked by LLM so only require phone_number
        if re.search(r"\b(yes|confirm|ඔව්|yep|හරි|ස්ථිර|ok|okay)\b", user_msg, re.IGNORECASE):
            if session.phone_number and not session.confirmed:
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
