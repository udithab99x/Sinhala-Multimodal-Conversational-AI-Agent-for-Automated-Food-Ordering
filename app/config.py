"""App configuration loaded from .env"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_from: str = "whatsapp:+14155238886"
    twilio_voice_from: str = ""

    # HuggingFace — the fine-tuned Gemma 4 model
    hf_token: str = ""
    hf_model_id: str = "google/gemma-3-4b-it"   # swap to your fine-tuned ID after training

    # Google Cloud
    google_application_credentials: str = ""

    # App
    base_url: str = "http://localhost:8000"
    port: int = 8000

    # Paths
    data_dir: Path = Path(__file__).parent.parent / "data"
    menu_path: Path = data_dir / "menu_data.json"
    audio_dir: Path = Path(__file__).parent.parent / "audio_cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
settings.audio_dir.mkdir(exist_ok=True)
