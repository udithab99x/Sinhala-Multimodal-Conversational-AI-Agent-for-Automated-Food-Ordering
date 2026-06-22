"""
Text-to-Speech using Google AI Studio REST API (Gemini TTS).
Works with a GOOGLE_API_KEY — no service account needed.
Falls back to gTTS if API key is not set.
"""

import hashlib
import os
import struct
import wave
import io
import base64
import logging
from pathlib import Path

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Gemini TTS endpoint — gemini-3.1-flash-tts-preview supports Sinhala auto-detection
_GEMINI_TTS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.1-flash-tts-preview:generateContent"
)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes into a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _synthesize_gemini(text: str, api_key: str) -> bytes:
    """Call Gemini TTS REST API, return MP3 bytes (via WAV intermediate)."""
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Aoede"}
                }
            },
        },
    }
    resp = requests.post(
        f"{_GEMINI_TTS_URL}?key={api_key}",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract inline audio data
    parts = data["candidates"][0]["content"]["parts"]
    audio_b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
    audio_bytes = base64.b64decode(audio_b64)

    # Gemini returns raw PCM (L16, 24kHz, mono) — wrap in WAV then convert to MP3
    wav_bytes = _pcm_to_wav(audio_bytes)

    # Try to convert WAV → MP3 via pydub/ffmpeg; fall back to returning WAV
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        mp3_buf = io.BytesIO()
        seg.export(mp3_buf, format="mp3")
        return mp3_buf.getvalue()
    except Exception:
        # Return WAV — Twilio can play WAV too
        return wav_bytes


def _synthesize_gtts(text: str) -> bytes:
    """Fallback: gTTS (free Google Translate TTS, no Sinhala but works for English)."""
    from gtts import gTTS
    buf = io.BytesIO()
    tts = gTTS(text=text, lang="si", slow=False)
    tts.write_to_fp(buf)
    return buf.getvalue()


class SinhalaTTS:
    """Synthesize Sinhala text to audio. Caches results by MD5 of text."""

    def __init__(self):
        self._cache_dir = settings.audio_dir
        self._api_key = settings.google_api_key
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str) -> Path:
        """Convert text to speech. Returns path to cached audio file."""
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        # Try MP3 first, then WAV (Gemini fallback)
        for ext in ("mp3", "wav"):
            cached = self._cache_dir / f"{cache_key}.{ext}"
            if cached.exists():
                return cached

        out_path = self._cache_dir / f"{cache_key}.mp3"
        try:
            if self._api_key:
                logger.info(f"[TTS] Gemini TTS: {text[:60]}...")
                audio = _synthesize_gemini(text, self._api_key)
                # Detect if we got WAV back (pydub unavailable)
                if audio[:4] == b"RIFF":
                    out_path = self._cache_dir / f"{cache_key}.wav"
            else:
                logger.warning("[TTS] No GOOGLE_API_KEY — falling back to gTTS")
                audio = _synthesize_gtts(text)
            out_path.write_bytes(audio)
        except Exception as e:
            logger.error(f"[TTS] Synthesis failed: {e}. Using silence fallback.")
            # Write 1 second of silence as WAV
            silence = _pcm_to_wav(b"\x00\x00" * 24000)
            out_path = self._cache_dir / f"{cache_key}.wav"
            out_path.write_bytes(silence)

        return out_path

    def get_public_url(self, text: str) -> str:
        """Synthesize and return a public URL Twilio can <Play>."""
        audio_path = self.synthesize(text)
        return f"{settings.base_url}/audio/{audio_path.name}"


_tts: SinhalaTTS | None = None


def get_tts() -> SinhalaTTS:
    global _tts
    if _tts is None:
        _tts = SinhalaTTS()
    return _tts
