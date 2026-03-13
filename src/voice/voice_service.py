"""
Voice Service -- STT (Deepgram) and TTS (ElevenLabs / Edge TTS fallback).

WHAT: Converts audio to text via Deepgram Nova-2, and text to audio via
      ElevenLabs (primary) or Edge TTS (free fallback). Includes text-cleaning
      logic that strips roleplay markup (*actions*, scene headers) while
      preserving ElevenLabs paralinguistic cues like (sighs) and (laughs).

WHY:  Voice is an optional but high-impact modality. ElevenLabs produces
      natural-sounding speech with emotional cues; Edge TTS is a zero-cost
      fallback for when the ElevenLabs quota is exhausted or the key isn't set.

HOW:  STT: POST raw audio to Deepgram's REST API, get back a transcript.
      TTS: Call ElevenLabs stream endpoint with the companion's configured
      voice_id and voice settings (stability, similarity, style). If
      ElevenLabs fails or isn't configured, fall back to Edge TTS (Microsoft's
      free neural TTS). Text cleaning removes asterisk-wrapped actions and
      markdown but keeps parenthetical cues for ElevenLabs.

Config: Voice identity (voice_id, settings) comes from persona_config.yaml.
"""

import os
import re
import asyncio
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Feature flag
VOICE_ENABLED = os.environ.get('COMPANION_VOICE_ENABLED', os.environ.get('COMPANION_VOICE_ENABLED', 'true')).lower() == 'true'
DEEPGRAM_API_KEY = os.environ.get('DEEPGRAM_API_KEY')
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')

# Voice identity from persona config (env vars override YAML)
def _get_voice_config():
    from src.config.persona_config import get_persona_config
    return get_persona_config()

_vc = _get_voice_config()
EDGE_TTS_VOICE = _vc.edge_tts_fallback
ELEVENLABS_VOICE_ID = _vc.elevenlabs_voice_id
ELEVENLABS_MODEL = _vc.elevenlabs_model


class VoiceService:
    """Singleton service for STT and TTS operations."""

    def __init__(self):
        self._deepgram_client = None
        self._elevenlabs_client = None
        self._initialized = False
        self._tts_engine: str = 'none'  # 'elevenlabs', 'edge_tts', or 'none'
        self._init_deepgram()
        self._init_tts()

    def _init_deepgram(self):
        """Initialize Deepgram client if API key is available."""
        if not DEEPGRAM_API_KEY:
            logger.warning("DEEPGRAM_API_KEY not set - STT disabled")
            return

        try:
            from deepgram import DeepgramClient
            self._deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
            self._initialized = True
            logger.info("Deepgram STT initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Deepgram client: {e}")

    def _init_tts(self):
        """Initialize TTS engine — ElevenLabs if available, else Edge TTS."""
        if ELEVENLABS_API_KEY:
            try:
                from elevenlabs.client import ElevenLabs
                self._elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                self._tts_engine = 'elevenlabs'
                logger.info(f"TTS engine: ElevenLabs (voice_id={ELEVENLABS_VOICE_ID}, model={ELEVENLABS_MODEL})")
            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs: {e}")
                self._tts_engine = 'edge_tts'
                logger.info("TTS engine: Edge TTS (ElevenLabs init failed, falling back)")
        else:
            self._tts_engine = 'edge_tts'
            logger.info(f"TTS engine: Edge TTS ({EDGE_TTS_VOICE}) — set ELEVENLABS_API_KEY for expressive voice")

    def is_enabled(self) -> bool:
        """Check if voice processing is available."""
        return VOICE_ENABLED and self._initialized

    def get_tts_engine(self) -> str:
        """Return which TTS engine is active: 'elevenlabs', 'edge_tts', or 'none'."""
        return self._tts_engine

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe an audio file to text using Deepgram Nova-2.

        Args:
            audio_file_path: Path to audio file (.ogg, .mp3, .wav, etc.)

        Returns:
            Transcribed text or None on failure.
        """
        if not self._deepgram_client:
            logger.error("Deepgram client not initialized")
            return None

        try:
            with open(audio_file_path, 'rb') as f:
                buffer_data = f.read()

            # Deepgram SDK v5: keyword args directly, no options class
            response = self._deepgram_client.listen.v1.media.transcribe_file(
                request=buffer_data,
                model="nova-2",
                smart_format=True,
                language="en",
            )

            # Extract transcript from response
            transcript = (
                response.results
                .channels[0]
                .alternatives[0]
                .transcript
            )

            if transcript:
                logger.info(f"Transcribed audio: {transcript[:80]}...")
                return transcript.strip()
            else:
                logger.warning("Deepgram returned empty transcript")
                return None

        except Exception as e:
            logger.error(f"STT transcription failed: {e}", exc_info=True)
            return None

    @staticmethod
    def clean_text_for_tts(text: str, allow_paralinguistics: bool = False) -> str:
        """
        Strip non-speakable markup from text before TTS.

        For Edge TTS (allow_paralinguistics=False):
            Removes *action text*, (paralinguistic cues), emojis, markdown, etc.
        For ElevenLabs (allow_paralinguistics=True):
            Keeps (sighs), (laughs) etc. but strips *action text*, emojis, markdown.
        """
        # Always remove *asterisk actions* like *smiles softly*, *leans against counter*
        # These are roleplay stage directions, not speakable text.
        # ElevenLabs paralinguistics use (parentheses) instead, handled below.
        text = re.sub(r'\*[^*]+\*', '', text)

        # Remove (parenthetical cues) only for non-ElevenLabs engines
        # ElevenLabs can actually perform (sighs), (laughs), (whispers), etc.
        if not allow_paralinguistics:
            text = re.sub(r'\([^)]+\)', '', text)

        # Remove emoji (Unicode emoji ranges)
        text = re.sub(
            r'[\U0001F600-\U0001F64F'   # emoticons
            r'\U0001F300-\U0001F5FF'     # symbols & pictographs
            r'\U0001F680-\U0001F6FF'     # transport & map
            r'\U0001F1E0-\U0001F1FF'     # flags
            r'\U00002702-\U000027B0'     # dingbats
            r'\U0000FE00-\U0000FE0F'     # variation selectors
            r'\U0001F900-\U0001F9FF'     # supplemental symbols
            r'\U0001FA00-\U0001FA6F'     # chess symbols
            r'\U0001FA70-\U0001FAFF'     # symbols extended
            r'\U00002600-\U000026FF'     # misc symbols
            r'\U0000200D'               # zero-width joiner
            r'\U00002764'               # heart
            r']+', '', text
        )

        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s', '', text)      # headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # bold
        text = re.sub(r'__([^_]+)__', r'\1', text)      # bold alt
        text = re.sub(r'~~([^~]+)~~', r'\1', text)      # strikethrough
        text = re.sub(r'`([^`]+)`', r'\1', text)        # inline code

        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)

        # Collapse multiple spaces/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)

        return text.strip()

    def synthesize_speech(self, text: str, output_path: str) -> bool:
        """
        Convert text to speech → OGG Opus for Telegram.

        Uses ElevenLabs if available, otherwise Edge TTS.
        Cleans text of non-speakable markup before synthesis.

        Args:
            text: Text to speak.
            output_path: Where to save the .ogg file.

        Returns:
            True if successful, False otherwise.
        """
        if self._tts_engine == 'elevenlabs':
            text = self.clean_text_for_tts(text, allow_paralinguistics=True)
            return self._synthesize_elevenlabs(text, output_path)
        else:
            text = self.clean_text_for_tts(text, allow_paralinguistics=False)
            return self._synthesize_edge_tts(text, output_path)

    def _synthesize_elevenlabs(self, text: str, output_path: str) -> bool:
        """Synthesize speech using ElevenLabs API."""
        mp3_path = None
        try:
            from pydub import AudioSegment

            mp3_path = output_path + '.mp3'

            # Generate audio via ElevenLabs
            audio_generator = self._elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL,
                output_format="mp3_44100_128",
            )

            # Write the generator output to file
            with open(mp3_path, 'wb') as f:
                for chunk in audio_generator:
                    f.write(chunk)

            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                logger.error("ElevenLabs produced empty output")
                return False

            # Convert mp3 → ogg opus (Telegram voice note format)
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(output_path, format='ogg', codec='libopus')

            logger.info(f"TTS [ElevenLabs]: {len(text)} chars → {os.path.getsize(output_path)} bytes ogg")
            return True

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}", exc_info=True)
            # Fall back to Edge TTS on failure
            logger.info("Falling back to Edge TTS...")
            return self._synthesize_edge_tts(text, output_path)
        finally:
            if mp3_path and os.path.exists(mp3_path):
                try:
                    os.unlink(mp3_path)
                except OSError:
                    pass

    def _synthesize_edge_tts(self, text: str, output_path: str) -> bool:
        """Synthesize speech using Edge TTS (free fallback)."""
        mp3_path = None
        try:
            from pydub import AudioSegment

            mp3_path = output_path + '.mp3'

            # Run the async edge_tts in a new event loop
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._edge_tts_save(text, mp3_path))
            finally:
                loop.close()

            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                logger.error("Edge TTS produced empty output")
                return False

            # Convert mp3 → ogg opus (Telegram voice note format)
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(output_path, format='ogg', codec='libopus')

            logger.info(f"TTS [Edge]: {len(text)} chars → {os.path.getsize(output_path)} bytes ogg")
            return True

        except Exception as e:
            logger.error(f"Edge TTS failed: {e}", exc_info=True)
            return False
        finally:
            if mp3_path and os.path.exists(mp3_path):
                try:
                    os.unlink(mp3_path)
                except OSError:
                    pass

    @staticmethod
    async def _edge_tts_save(text: str, output_path: str):
        """Async helper to run edge_tts.Communicate.save()."""
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        await communicate.save(output_path)


# Singleton
_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """Get singleton VoiceService instance."""
    global _service
    if _service is None:
        _service = VoiceService()
    return _service
