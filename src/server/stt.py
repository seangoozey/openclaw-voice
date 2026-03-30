"""
Speech-to-Text module using Whisper.
"""

import asyncio
import os

import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self._backend = "mock"
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel

            if self.device == "auto":
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                else:
                    self.device = "cpu"

            target_device = self.device if self.device != "mps" else "cpu"
            for compute_type in self._compute_type_candidates():
                try:
                    logger.info(
                        f"Loading faster-whisper {self.model_name} on {self.device} "
                        f"(compute_type={compute_type})"
                    )
                    self.model = WhisperModel(
                        self.model_name,
                        device=target_device,
                        compute_type=compute_type,
                    )
                    self._backend = "faster-whisper"
                    logger.info("faster-whisper loaded")
                    return
                except Exception as e:
                    logger.warning(
                        f"faster-whisper failed with compute_type={compute_type}: {e}"
                    )
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")

        try:
            import whisper

            if self.device == "auto":
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")

        logger.warning("No STT backend - using mock mode")
        self._backend = "mock"

    def _compute_type_candidates(self) -> list[str]:
        """Choose safer faster-whisper compute types for the current device."""
        requested = os.environ.get("OPENCLAW_STT_COMPUTE_TYPE", "").strip().lower()
        if requested:
            return [requested]

        if self.device == "cuda":
            # Pascal-era cards often prefer mixed/int8 or full float32 over float16.
            return ["int8_float16", "float32", "float16", "int8"]

        return ["int8"]

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()

        if self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()

        logger.debug(f"Mock STT: received {len(audio)} samples")
        return "[Mock transcription - install whisper for real STT]"
