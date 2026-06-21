"""
Google Cloud Text-to-Speech wrapper — Sinhala (si-LK) voice synthesis.
Used for voice call responses via Twilio <Play>.
"""

import hashlib
from pathlib import Path
from google.cloud import texttospeech
from app.config import settings


class SinhalaTTS:
    """Synthesize Sinhala text to audio. Caches results to avoid re-generating."""

    _VOICE = texttospeech.VoiceSelectionParams(
        language_code="si-LK",
        name="si-LK-Standard-A",   # Sinhala female voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    _AUDIO_CONFIG = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.9,   # slightly slower for clarity
    )

    def __init__(self):
        self._client = texttospeech.TextToSpeechClient()
        self._cache_dir = settings.audio_dir

    def synthesize(self, text: str) -> Path:
        """
        Convert text to Sinhala speech.
        Returns the path to the cached MP3 file.
        """
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        out_path = self._cache_dir / f"{cache_key}.mp3"

        if out_path.exists():
            return out_path

        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self._client.synthesize_speech(
            input=synthesis_input,
            voice=self._VOICE,
            audio_config=self._AUDIO_CONFIG,
        )
        out_path.write_bytes(response.audio_content)
        return out_path

    def get_public_url(self, text: str) -> str:
        """Synthesize and return a public URL Twilio can <Play>."""
        audio_path = self.synthesize(text)
        filename = audio_path.name
        return f"{settings.base_url}/audio/{filename}"


_tts: SinhalaTTS | None = None


def get_tts() -> SinhalaTTS:
    global _tts
    if _tts is None:
        _tts = SinhalaTTS()
    return _tts
