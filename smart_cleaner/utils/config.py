"""
Configuration management for Smart Cleaner.
Handles API keys, settings, and environment variables.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class for Smart Cleaner."""

    gemini_api_key: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Ollama settings (local LLM - FREE!)
    ollama_model: str = "llama3.2"
    ollama_base_url: Optional[str] = None  # Default: http://localhost:11434

    def __post_init__(self):
        """Load configuration from environment if not provided."""
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.ollama_base_url:
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if not self.ollama_model or self.ollama_model == "llama3.2":
            env_model = os.getenv("OLLAMA_MODEL")
            if env_model:
                self.ollama_model = env_model

    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.gemini_api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY environment variable or pass api_key parameter."
            )
        return True


# Default configuration instance
default_config = Config()
