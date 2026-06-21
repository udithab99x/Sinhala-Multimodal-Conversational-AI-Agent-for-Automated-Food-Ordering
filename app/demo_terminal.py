"""
Terminal demo — run the food ordering agent locally without Twilio or Google Cloud.
Uses the fine-tuned Gemma 4 model directly.

Usage:
    python app/demo_terminal.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import FoodOrderingAgent
from app.config import settings

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     Wakwalle Kade — Food Ordering AI Demo                    ║
║     Model: Gemma 4 E2B (Fine-tuned)  |  Pickup Only         ║
║     Type 'quit' to exit | 'reset' to start new session      ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    print("Loading model... (first run may take a few minutes)\n")

    agent = FoodOrderingAgent(
        menu_path=settings.menu_path,
        model_id=settings.hf_model_id,
        hf_token=settings.hf_token,
    )

    user_id = "demo_user"
    print("Agent: ආයුබෝවන්! Wakwalle Kade. ඔබට කුමක් ඕනේද? (Pickup only)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Agent: ස්තූතියි! ආයෙ contact කරන්න. Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset_session(user_id)
            print("Agent: New session started. ආයුබෝවන්! ඔබට කුමක් ඕනේද?\n")
            continue

        reply = agent.respond(user_id=user_id, user_message=user_input)
        print(f"Agent: {reply}\n")

        # Show session state (debug)
        session = agent.get_session(user_id)
        if session and session.confirmed:
            print(f"\n{'='*50}")
            print("ORDER CONFIRMED!")
            print(session.order_summary())
            print(f"{'='*50}\n")
            agent.reset_session(user_id)
            print("Agent: ආයුබෝවන්! නව order කිරීමට කතා කරන්න.\n")


if __name__ == "__main__":
    main()
