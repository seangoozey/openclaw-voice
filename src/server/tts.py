"""
Text-to-Speech module using OpenAI-compatible APIs, ElevenLabs, or local fallbacks.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger


class ChatterboxTTS:
    """Text-to-Speech using OpenAI-compatible APIs, ElevenLabs, Chatterbox, or fallbacks."""

    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        response_format: str = "pcm",
        sample_rate: int = 24000,
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"
        self.sample_rate = sample_rate

        self.model = None
        self._backend = "mock"
        self._preferred_backend = (
            backend or os.environ.get("OPENCLAW_TTS_BACKEND", "")
        ).strip().lower() or None
        self._openai_client = None
        self._elevenlabs_client = None

        self._tts_api_key = (
            api_key
            or os.environ.get("OPENCLAW_TTS_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self._tts_base_url = (
            base_url
            or os.environ.get("OPENCLAW_TTS_API_BASE_URL")
            or os.environ.get("OPENCLAW_TTS_BASE_URL")
            or ""
        ).rstrip("/")
        self._tts_model = model or os.environ.get("OPENCLAW_TTS_API_MODEL") or "gpt-4o-mini-tts"
        self._tts_voice = voice or os.environ.get("OPENCLAW_TTS_API_VOICE") or "alloy"
        self._tts_response_format = (
            response_format or os.environ.get("OPENCLAW_TTS_RESPONSE_FORMAT") or "pcm"
        )

        self._load_model()

    def _load_model(self):
        """Load the preferred TTS model, then fall back."""
        preferred = self._preferred_backend

        if preferred in {"openai", "openai-compatible"}:
            if self._try_openai_compatible_tts():
                return
            logger.warning("OpenAI-compatible TTS requested but unavailable, falling back")

        if preferred in {None, "", "elevenlabs"} and self._try_elevenlabs_tts():
            return

        if preferred in {None, "", "openai", "openai-compatible"} and self._try_openai_compatible_tts():
            return

        try:
            from chatterbox.tts import ChatterboxTTS as CBModel

            logger.info("Loading Chatterbox TTS...")
            self.model = CBModel.from_pretrained(device=self._get_device())
            self._backend = "chatterbox"
            logger.info("Chatterbox TTS ready")
            return
        except ImportError:
            logger.warning("Chatterbox not installed")
        except Exception as e:
            logger.warning(f"Chatterbox failed: {e}")

        try:
            from TTS.api import TTS

            logger.info("Loading Coqui XTTS...")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._backend = "xtts"
            logger.info("XTTS ready")
            return
        except ImportError:
            logger.warning("Coqui TTS not installed")
        except Exception as e:
            logger.warning(f"XTTS failed: {e}")

        logger.warning("No TTS backend available, using mock mode (silence)")
        self._backend = "mock"

    def _try_openai_compatible_tts(self) -> bool:
        """Initialize an OpenAI-compatible speech backend."""
        if not self._tts_base_url:
            return False

        try:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                api_key=self._tts_api_key or "unused",
                base_url=self._tts_base_url,
            )
            self._backend = "openai"
            logger.info(
                f"OpenAI-compatible TTS ready (base_url={self._tts_base_url}, model={self._tts_model})"
            )
            return True
        except ImportError:
            logger.error("openai package not installed")
        except Exception as e:
            logger.warning(f"OpenAI-compatible TTS failed: {e}")

        return False

    def _try_elevenlabs_tts(self) -> bool:
        """Initialize ElevenLabs if configured."""
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if not elevenlabs_key:
            return False

        try:
            from elevenlabs import ElevenLabs

            self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
            self._backend = "elevenlabs"
            logger.info("ElevenLabs TTS ready")
            return True
        except ImportError:
            logger.warning("ElevenLabs SDK not installed")
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}")

        return False

    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text."""
        if self._backend == "openai":
            return await self._synthesize_openai(text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.

        Yields raw PCM audio chunks.
        """
        if self._backend == "openai":
            try:
                async with self._openai_client.audio.speech.with_streaming_response.create(
                    model=self._tts_model,
                    voice=self._tts_voice,
                    input=text,
                    response_format=self._tts_response_format,
                ) as response:
                    async for chunk in response.iter_bytes():
                        if chunk:
                            yield chunk
            except Exception as e:
                logger.error(f"OpenAI-compatible TTS streaming error: {e}")
            return

        if self._backend == "elevenlabs":
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                for chunk in audio_generator:
                    yield chunk
            except Exception as e:
                logger.error(f"ElevenLabs streaming error: {e}")
            return

        audio = await self.synthesize(text)
        yield self._float_audio_to_pcm(audio)

    async def _synthesize_openai(self, text: str) -> np.ndarray:
        """Synthesize speech via an OpenAI-compatible endpoint."""
        try:
            response = await self._openai_client.audio.speech.create(
                model=self._tts_model,
                voice=self._tts_voice,
                input=text,
                response_format=self._tts_response_format,
            )
            audio_bytes = await response.aread()
            return self._pcm_bytes_to_float_audio(audio_bytes)
        except Exception as e:
            logger.error(f"OpenAI-compatible TTS error: {e}")
            return np.zeros(self.sample_rate, dtype=np.float32)

    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis."""
        if self._backend == "elevenlabs":
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                audio_bytes = b"".join(audio_generator)
                return self._pcm_bytes_to_float_audio(audio_bytes)
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
                return np.zeros(self.sample_rate, dtype=np.float32)

        if self._backend == "chatterbox":
            if self.voice_sample:
                audio = self.model.generate(text, audio_prompt=self.voice_sample)
            else:
                audio = self.model.generate(text)
            return audio.cpu().numpy().astype(np.float32)

        if self._backend == "xtts":
            if self.voice_sample:
                wav = self.model.tts(text=text, speaker_wav=self.voice_sample, language="en")
            else:
                wav = self.model.tts(text=text, language="en")
            return np.array(wav, dtype=np.float32)

        logger.debug(f"Mock TTS: '{text[:50]}...'")
        return np.zeros(self.sample_rate // 2, dtype=np.float32)

    def _pcm_bytes_to_float_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Convert 16-bit PCM bytes to float32 audio."""
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_array.astype(np.float32) / 32768.0

    def _float_audio_to_pcm(self, audio: np.ndarray) -> bytes:
        """Convert float32 audio to 16-bit PCM bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()
