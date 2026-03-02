import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))

def get_settings() -> Settings:
    return Settings()
