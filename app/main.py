"""
FastAPI application — entry point for all webhooks and REST endpoints.

Endpoints:
  POST /whatsapp/incoming   — Twilio WhatsApp webhook
  POST /voice/incoming      — Twilio voice call (greets caller, starts gather)
  POST /voice/process       — Twilio speech gather callback
  GET  /audio/{filename}    — Serve TTS audio files for Twilio <Play>
  GET  /health              — Health check
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import Response, FileResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather

from app.agent import get_agent
from app.tts import get_tts
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sinhala Food Ordering AI", version="1.0")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.hf_model_id}


# ── Serve TTS audio ───────────────────────────────────────────────────────────

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    audio_path = settings.audio_dir / filename
    if not audio_path.exists():
        return Response(status_code=404)
    return FileResponse(str(audio_path), media_type="audio/mpeg")


# ── WhatsApp webhook ──────────────────────────────────────────────────────────

@app.post("/whatsapp/incoming")
async def whatsapp_incoming(
    From: str = Form(...),       # e.g. whatsapp:+94771234567
    Body: str = Form(...),
):
    """
    Twilio sends an HTTP POST here when a WhatsApp message arrives.
    We respond with TwiML MessagingResponse.
    """
    user_id = From.replace("whatsapp:", "").strip()
    logger.info(f"WhatsApp from {user_id}: {Body}")

    agent = get_agent()
    reply = agent.respond(user_id=user_id, user_message=Body)
    logger.info(f"Reply to {user_id}: {reply}")

    twiml = MessagingResponse()
    twiml.message(reply)
    return Response(content=str(twiml), media_type="application/xml")


# ── Voice call webhooks ───────────────────────────────────────────────────────

GREETING_SI = (
    "ආයුබෝවන්! Wakwalle Kade ඇණවුම් AI assistant. "
    "ඔබට කුමක් ඕනේද? Pickup only. කරුණාකර කතා කරන්න."
)


@app.post("/voice/incoming")
async def voice_incoming():
    """
    Called when a customer dials the Twilio number.
    Greet in Sinhala, then start listening.
    """
    tts = get_tts()
    greeting_url = tts.get_public_url(GREETING_SI)

    resp = VoiceResponse()
    gather = Gather(
        input="speech",
        language="si-LK",            # Sinhala STT via Twilio + Google
        speech_timeout="auto",
        action=f"{settings.base_url}/voice/process",
        method="POST",
    )
    gather.play(greeting_url)
    resp.append(gather)

    # Fallback if caller says nothing
    resp.redirect(f"{settings.base_url}/voice/incoming")
    return Response(content=str(resp), media_type="application/xml")


@app.post("/voice/process")
async def voice_process(
    From: str = Form(...),
    SpeechResult: str = Form(default=""),
    Confidence: float = Form(default=0.0),
):
    """
    Called after Twilio captures speech and transcribes it.
    SpeechResult = the transcript of what the caller said.
    """
    user_id = From.strip()
    logger.info(f"Voice from {user_id}: '{SpeechResult}' (confidence {Confidence:.2f})")

    resp = VoiceResponse()

    if not SpeechResult.strip():
        # Could not transcribe — ask again
        tts = get_tts()
        sorry_url = tts.get_public_url("කනගාටුයි, නැවත කතා කරන්නද?")
        resp.play(sorry_url)
        resp.redirect(f"{settings.base_url}/voice/incoming")
        return Response(content=str(resp), media_type="application/xml")

    # Get agent response
    agent = get_agent()
    reply_text = agent.respond(user_id=user_id, user_message=SpeechResult)
    logger.info(f"Agent reply: {reply_text}")

    # Synthesize reply
    tts = get_tts()
    reply_url = tts.get_public_url(reply_text)

    # Play reply then listen again
    gather = Gather(
        input="speech",
        language="si-LK",
        speech_timeout="auto",
        action=f"{settings.base_url}/voice/process",
        method="POST",
    )
    gather.play(reply_url)
    resp.append(gather)
    resp.redirect(f"{settings.base_url}/voice/incoming")

    return Response(content=str(resp), media_type="application/xml")


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
