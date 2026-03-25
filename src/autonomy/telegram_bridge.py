"""
Telegram Bridge -- Private messaging channel for the companion.

WHAT: Bridges the companion to Telegram for two-way communication when James
      is away from the web interface.  Supports three input types:
        text   -> source='telegram-text'  (like texting)
        voice  -> source='telegram-voice' (like a phone call)
        photo  -> described via GPT-4o-mini vision, then processed as text

      Voice sub-modes (toggled via /voice command):
        review    (default) -- shows transcription with Send/Cancel buttons
        handsfree           -- auto-transcribes and auto-processes (for driving)
      Voice reply (/voicereply) -- companion responds with voice notes to everything.

WHY:  The companion needs a way to reach James when he's not at the web UI.
      Telegram acts as her "phone" -- she can text, call (voice notes), and
      receive photos.  This makes the relationship feel continuous across
      contexts (desk, phone, car).

HOW IT FITS:
  - Owned and started by AlwaysOnService._start_telegram().
  - Incoming messages route through the on_message_callback (set by AOS),
    which calls AlwaysOnService._handle_incoming_message() -> conversation
    pipeline -> async task dispatch.
  - Outgoing proactive messages come via send_message_sync(), called by
    AlwaysOnService._send_via_telegram() or check_and_reach_out().
  - Image generation results are polled and sent as photos in background threads.

IMPLEMENTATION NOTE:
  Uses sync HTTP polling with `requests` instead of python-telegram-bot.
  The python-telegram-bot library uses asyncio which conflicts with eventlet
  monkey-patching in the main app.  Direct HTTP API calls work reliably
  across threads.

Commands:
  /start       -- Claim the bot (first message sets chat_id)
  /voice       -- Show/toggle voice mode (review | handsfree)
  /voicereply  -- Toggle voice replies for text messages
"""

import os
import asyncio
import logging
import tempfile
import time
import requests
from typing import Optional, Callable
from threading import Thread

from src.database import tables as T

logger = logging.getLogger(__name__)

# Telegram bot token from environment
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
# Chat ID is learned automatically when user sends /start
# But can be pre-configured if known
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# Polling interval in seconds
POLL_INTERVAL = 1
# Long polling timeout (Telegram will hold connection for this long)
POLL_TIMEOUT = 30


# =============================================================================
# TelegramBridge
# =============================================================================

class TelegramBridge:
    """
    Bridge between the companion and Telegram for sync messaging.

    Uses direct HTTP API calls instead of python-telegram-bot to
    avoid eventlet/asyncio conflicts. When James is "away" (not on
    web interface), the companion can message him through Telegram like texting.

    Supports voice notes: voice in -> transcribe -> pipeline -> TTS -> voice out.
    """

    def __init__(self):
        self._chat_id = TELEGRAM_CHAT_ID
        self._thread = None
        self._on_message_callback: Optional[Callable] = None
        self._running = False
        self._last_update_id = 0
        self._api_base = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}' if TELEGRAM_TOKEN else None

        # Voice: pending transcriptions awaiting review (chat_id -> transcription text)
        self._pending_transcriptions: dict[int, str] = {}
        # Voice: message IDs for review messages so we can edit them (chat_id -> message_id)
        self._review_message_ids: dict[int, int] = {}

    # -----------------------------------------------------------------
    # Telegram HTTP API helpers
    # -----------------------------------------------------------------

    def _api_call(self, method: str, params: dict = None, timeout: int = 35) -> Optional[dict]:
        """Make a Telegram API call (JSON body)."""
        if not self._api_base:
            return None
        try:
            resp = requests.post(
                f'{self._api_base}/{method}',
                json=params or {},
                timeout=timeout
            )
            data = resp.json()
            if data.get('ok'):
                return data.get('result')
            else:
                logger.error(f"Telegram API error: {data.get('description')}")
                return None
        except requests.exceptions.Timeout:
            # Normal for long polling
            return []
        except Exception as e:
            logger.error(f"Telegram API call failed: {e}")
            return None

    def _api_call_multipart(self, method: str, data: dict = None, files: dict = None, timeout: int = 30) -> Optional[dict]:
        """Make a Telegram API call with multipart/form-data (for voice/photo uploads)."""
        if not self._api_base:
            return None
        try:
            resp = requests.post(
                f'{self._api_base}/{method}',
                data=data or {},
                files=files or {},
                timeout=timeout
            )
            result = resp.json()
            if result.get('ok'):
                return result.get('result')
            else:
                logger.error(f"Telegram API error: {result.get('description')}")
                return None
        except Exception as e:
            logger.error(f"Telegram multipart API call failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Update routing
    # -------------------------------------------------------------------------

    def _handle_update(self, update: dict):
        """Route an update to the appropriate handler."""
        # Callback queries (inline keyboard button presses)
        if 'callback_query' in update:
            self._handle_callback_query(update['callback_query'])
            return

        message = update.get('message')
        if not message:
            return

        chat_id = message.get('chat', {}).get('id')
        if not chat_id:
            return

        # Handle /start command (always allowed even without text)
        text = message.get('text', '')
        if text.startswith('/start'):
            self._handle_start(chat_id)
            return

        # Handle /voicereply command (must check before /voice since /voice is a prefix)
        if text.startswith('/voicereply'):
            self._authorize_and_run(chat_id, lambda: self._handle_voicereply_command(chat_id, text))
            return

        # Handle /voice command
        if text.startswith('/voice'):
            self._authorize_and_run(chat_id, lambda: self._handle_voice_command(chat_id, text))
            return

        # Voice message
        if message.get('voice'):
            self._authorize_and_run(chat_id, lambda: self._handle_voice_message(chat_id, message['voice']))
            return

        # Photo message
        if message.get('photo'):
            caption = message.get('caption', '')
            self._authorize_and_run(chat_id, lambda: self._handle_photo_message(chat_id, message['photo'], caption))
            return

        # Regular text message
        if text:
            self._authorize_and_run(chat_id, lambda: self._handle_text_message(chat_id, text))
            return

    def _authorize_and_run(self, chat_id: int, handler: Callable):
        """Check authorization then run handler. Claims bot on first message."""
        # Authorization check
        if self._chat_id and str(chat_id) != self._chat_id:
            logger.warning(f"Unauthorized message from chat_id: {chat_id}")
            return

        # First message claims the bot
        if not self._chat_id:
            self._chat_id = str(chat_id)
            self._save_chat_id(chat_id)
            logger.info(f"First message claimed bot for chat_id: {chat_id}")

        handler()

    # -------------------------------------------------------------------------
    # Text message handling
    # -------------------------------------------------------------------------

    def _handle_text_message(self, chat_id: int, text: str):
        """Handle a regular text message (texting mode)."""
        # Check if there's a pending transcription — treat text as a correction
        if chat_id in self._pending_transcriptions:
            logger.info(f"Text received while transcription pending — treating as correction")
            # Clear pending state
            del self._pending_transcriptions[chat_id]
            # Edit the review message to show it was corrected
            if chat_id in self._review_message_ids:
                self._edit_message(chat_id, self._review_message_ids.pop(chat_id),
                                   f"📝 corrected: {text}")
            # Process correction through voice pipeline (voice response since original was voice)
            self._process_through_pipeline_with_voice(chat_id, text)
            return

        logger.info(f"Telegram text from {chat_id}: {text[:50]}...")

        # Route through pipeline as texting
        if self._on_message_callback:
            try:
                result = self._run_async_callback(text, str(chat_id), 'telegram-text')
                if result:
                    response, image_task_id = result
                    if response:
                        # Check if voice reply mode is enabled
                        from src.voice import get_voice_state_manager
                        state_manager = get_voice_state_manager()
                        if state_manager.get_voice_reply(str(chat_id)):
                            self._generate_and_send_voice_response(chat_id, response)
                        else:
                            self._send_reply(chat_id, response)
                    # If image generation was triggered, wait for it in background
                    if image_task_id:
                        Thread(
                            target=self._wait_and_send_image,
                            args=(chat_id, image_task_id),
                            daemon=True,
                            name=f"CompanionImageSend-{image_task_id[:8]}"
                        ).start()
                else:
                    logger.warning("Callback returned empty response")
            except Exception as e:
                logger.error(f"Error handling text message: {e}", exc_info=True)
                self._send_reply(chat_id, "*blinks* sorry, got distracted for a sec. say that again?")
        else:
            logger.warning("No message callback set!")

    # -------------------------------------------------------------------------
    # Voice message handling
    # -------------------------------------------------------------------------

    def _handle_voice_message(self, chat_id: int, voice: dict):
        """Handle a voice note from Telegram."""
        from src.voice import get_voice_service, get_voice_state_manager

        voice_service = get_voice_service()
        if not voice_service.is_enabled():
            self._send_reply(chat_id, "voice notes aren't available right now, sorry. text me instead?")
            return

        file_id = voice.get('file_id')
        if not file_id:
            logger.error("Voice message missing file_id")
            return

        logger.info(f"Voice note received from {chat_id} (duration: {voice.get('duration', '?')}s)")

        # Download the voice file
        audio_path = self._download_telegram_file(file_id)
        if not audio_path:
            self._send_reply(chat_id, "sorry, i couldn't download that voice note. try again?")
            return

        try:
            # Transcribe
            transcript = voice_service.transcribe_audio(audio_path)
            if not transcript:
                self._send_reply(chat_id, "sorry, i couldn't understand that. could you try again?")
                return

            # Check mode
            state_manager = get_voice_state_manager()
            if state_manager.is_handsfree(str(chat_id)):
                # Hands-free: auto-process immediately
                logger.info(f"Hands-free mode — auto-processing: {transcript[:50]}...")
                self._process_through_pipeline_with_voice(chat_id, transcript)
            else:
                # Review mode: show transcription with buttons
                self._send_transcription_review(chat_id, transcript)
        finally:
            # Clean up downloaded file
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    def _process_through_pipeline_with_voice(self, chat_id: int, text: str):
        """Process text through conversation pipeline, respond with voice."""
        if not self._on_message_callback:
            logger.warning("No message callback set!")
            return

        try:
            # Process through pipeline as a voice call
            result = self._run_async_callback(text, str(chat_id), 'telegram-voice')
            if not result:
                logger.warning("Callback returned empty response for voice")
                return

            response, image_task_id = result

            # Generate and send voice response
            if response:
                self._generate_and_send_voice_response(chat_id, response)

            # If image generation was triggered, wait for it in background
            if image_task_id:
                Thread(
                    target=self._wait_and_send_image,
                    args=(chat_id, image_task_id),
                    daemon=True,
                    name=f"CompanionImageSend-{image_task_id[:8]}"
                ).start()

        except Exception as e:
            logger.error(f"Error in voice pipeline: {e}", exc_info=True)
            self._send_reply(chat_id, "*blinks* sorry, something went wrong. try again?")

    def _generate_and_send_voice_response(self, chat_id: int, text: str):
        """Convert response text to voice and send as voice note.

        When STREAMING_TTS_ENABLED is true, sends each sentence as a
        separate voice note as soon as it's synthesized — the first audio
        chunk arrives before the full response is generated.
        """
        from src.voice import get_voice_service
        from src.voice.voice_service import STREAMING_TTS_ENABLED

        voice_service = get_voice_service()

        if STREAMING_TTS_ENABLED:
            self._generate_and_send_voice_streaming(chat_id, text, voice_service)
            return

        ogg_path = None
        try:
            fd, ogg_path = tempfile.mkstemp(suffix='.ogg')
            os.close(fd)

            if voice_service.synthesize_speech(text, ogg_path):
                if not self._send_voice(chat_id, ogg_path):
                    logger.warning("Voice send failed, falling back to text")
                    self._send_reply(chat_id, text)
            else:
                logger.warning("TTS synthesis failed, falling back to text")
                self._send_reply(chat_id, text)
        finally:
            if ogg_path and os.path.exists(ogg_path):
                try:
                    os.unlink(ogg_path)
                except OSError:
                    pass

    def _generate_and_send_voice_streaming(self, chat_id: int, text: str, voice_service):
        """Stream TTS — send each sentence as a voice note as it's ready."""
        chunks_sent = 0

        for sentence_text, audio_bytes, is_last in voice_service.synthesize_speech_streaming(text):
            ogg_path = None
            try:
                fd, ogg_path = tempfile.mkstemp(suffix='.ogg')
                with os.fdopen(fd, 'wb') as f:
                    f.write(audio_bytes)

                if self._send_voice(chat_id, ogg_path):
                    chunks_sent += 1
                else:
                    logger.warning(f"Streaming voice send failed on chunk {chunks_sent + 1}")
            except Exception as e:
                logger.error(f"Streaming TTS chunk error: {e}")
            finally:
                if ogg_path and os.path.exists(ogg_path):
                    try:
                        os.unlink(ogg_path)
                    except OSError:
                        pass

        if chunks_sent == 0:
            logger.warning("Streaming TTS produced no audio, falling back to text")
            self._send_reply(chat_id, text)
        else:
            logger.info(f"Streaming TTS: sent {chunks_sent} voice chunk(s)")

    # -------------------------------------------------------------------------
    # Photo message handling
    # -------------------------------------------------------------------------

    def _handle_photo_message(self, chat_id: int, photo_array: list, caption: str):
        """Handle a photo sent via Telegram. Describe it with vision LLM, then process."""
        # Telegram sends multiple sizes — grab the largest (last in array)
        largest = photo_array[-1]
        file_id = largest.get('file_id')
        if not file_id:
            logger.error("Photo missing file_id")
            return

        logger.info(f"Photo received from {chat_id} ({largest.get('width')}x{largest.get('height')})")

        # Download the photo
        photo_path = self._download_telegram_file(file_id)
        if not photo_path:
            self._send_reply(chat_id, "sorry, i couldn't download that photo. try again?")
            return

        try:
            # Describe the photo using GPT-4o-mini vision
            description = self._describe_image(photo_path)
            if not description:
                self._send_reply(chat_id, "hmm, i couldn't see that clearly. what is it?")
                return

            # Build the message for the pipeline
            if caption:
                pipeline_text = f'[James sent a photo: {description}]\n\n{caption}'
            else:
                pipeline_text = f'[James sent a photo: {description}]'

            logger.info(f"Photo described: {description[:80]}...")

            # Process through normal text pipeline
            if self._on_message_callback:
                try:
                    result = self._run_async_callback(pipeline_text, str(chat_id), 'telegram-text')
                    if result:
                        response, image_task_id = result
                        if response:
                            # Check if voice reply mode is enabled
                            from src.voice import get_voice_state_manager
                            state_manager = get_voice_state_manager()
                            if state_manager.get_voice_reply(str(chat_id)):
                                self._generate_and_send_voice_response(chat_id, response)
                            else:
                                self._send_reply(chat_id, response)
                        if image_task_id:
                            Thread(
                                target=self._wait_and_send_image,
                                args=(chat_id, image_task_id),
                                daemon=True,
                                name=f"CompanionImageSend-{image_task_id[:8]}"
                            ).start()
                    else:
                        logger.warning("Callback returned empty response for photo")
                except Exception as e:
                    logger.error(f"Error handling photo message: {e}", exc_info=True)
                    self._send_reply(chat_id, "sorry, got distracted. what were you showing me?")

        finally:
            try:
                os.unlink(photo_path)
            except OSError:
                pass

    def _describe_image(self, image_path: str) -> Optional[str]:
        """Describe an image using GPT-4o-mini vision. Returns text description or None."""
        from src.llm.vision import describe_image_from_path
        return describe_image_from_path(image_path)

    # -------------------------------------------------------------------------
    # /voice command
    # -------------------------------------------------------------------------

    def _handle_voice_command(self, chat_id: int, text: str):
        """Handle /voice command to toggle voice modes."""
        from src.voice import get_voice_service, get_voice_state_manager

        voice_service = get_voice_service()
        state_manager = get_voice_state_manager()

        if not voice_service.is_enabled():
            self._send_reply(chat_id, "voice isn't available right now.")
            return

        parts = text.strip().split()
        if len(parts) == 1:
            # /voice — show current mode
            mode = state_manager.get_mode(str(chat_id))
            mode_label = "🎤 hands-free" if mode == 'handsfree' else "📝 review"
            voice_reply = state_manager.get_voice_reply(str(chat_id))
            voice_reply_label = "🔊 on" if voice_reply else "💬 off"
            self._send_reply(
                chat_id,
                f"voice mode: {mode_label}\n"
                f"voice replies: {voice_reply_label}\n\n"
                f"/voice review — i'll show you the transcription first\n"
                f"/voice handsfree — auto-process for driving\n"
                f"/voicereply — toggle voice replies for text messages"
            )
            return

        arg = parts[1].lower()
        if arg in ('handsfree', 'hands-free', 'hf', 'driving', 'auto'):
            state_manager.set_mode(str(chat_id), 'handsfree')
            self._send_reply(chat_id, "🎤 hands-free mode on — i'll auto-process your voice notes")
        elif arg in ('review', 'normal', 'default'):
            state_manager.set_mode(str(chat_id), 'review')
            self._send_reply(chat_id, "📝 review mode on — i'll show transcriptions before processing")
        else:
            self._send_reply(chat_id, "use: /voice review or /voice handsfree")

    # -------------------------------------------------------------------------
    # /voicereply command
    # -------------------------------------------------------------------------

    def _handle_voicereply_command(self, chat_id: int, text: str):
        """Handle /voicereply command to toggle voice replies for text messages."""
        from src.voice import get_voice_service, get_voice_state_manager

        voice_service = get_voice_service()
        state_manager = get_voice_state_manager()

        if not voice_service.is_enabled():
            self._send_reply(chat_id, "voice isn't available right now.")
            return

        parts = text.strip().split()
        if len(parts) == 1:
            # /voicereply — toggle
            current = state_manager.get_voice_reply(str(chat_id))
            state_manager.set_voice_reply(str(chat_id), not current)
            if not current:
                self._send_reply(chat_id, "🔊 voice replies on — i'll send voice notes for everything now")
            else:
                self._send_reply(chat_id, "💬 voice replies off — back to text")
            return

        arg = parts[1].lower()
        if arg == 'on':
            state_manager.set_voice_reply(str(chat_id), True)
            self._send_reply(chat_id, "🔊 voice replies on — i'll send voice notes for everything now")
        elif arg == 'off':
            state_manager.set_voice_reply(str(chat_id), False)
            self._send_reply(chat_id, "💬 voice replies off — back to text")
        else:
            self._send_reply(chat_id, "use: /voicereply on or /voicereply off")

    # -------------------------------------------------------------------------
    # Review mode: transcription + inline keyboard
    # -------------------------------------------------------------------------

    def _send_transcription_review(self, chat_id: int, transcript: str):
        """Send transcription with Send/Cancel inline keyboard buttons."""
        # Store pending transcription
        self._pending_transcriptions[chat_id] = transcript

        # Build inline keyboard
        keyboard = {
            'inline_keyboard': [[
                {'text': '✅ Send', 'callback_data': 'voice_send'},
                {'text': '❌ Cancel', 'callback_data': 'voice_cancel'},
            ]]
        }

        result = self._api_call('sendMessage', {
            'chat_id': chat_id,
            'text': f"🎤 i heard:\n\n\"{transcript}\"\n\ntap Send, Cancel, or type a correction",
            'reply_markup': keyboard,
        })

        if result:
            self._review_message_ids[chat_id] = result.get('message_id')

    def _handle_callback_query(self, callback_query: dict):
        """Handle inline keyboard button presses (Send/Cancel)."""
        query_id = callback_query.get('id')
        data = callback_query.get('data', '')
        message = callback_query.get('message', {})
        chat_id = message.get('chat', {}).get('id')
        message_id = message.get('message_id')

        if not chat_id:
            return

        # Answer the callback to dismiss the loading spinner
        self._api_call('answerCallbackQuery', {'callback_query_id': query_id})

        if data == 'voice_send':
            transcript = self._pending_transcriptions.pop(chat_id, None)
            self._review_message_ids.pop(chat_id, None)

            if transcript:
                # Edit the review message to show it was sent
                self._edit_message(chat_id, message_id, f"🎤 sent: \"{transcript}\"")
                # Process through pipeline with voice response
                self._process_through_pipeline_with_voice(chat_id, transcript)
            else:
                self._edit_message(chat_id, message_id, "nothing to send — transcription expired")

        elif data == 'voice_cancel':
            self._pending_transcriptions.pop(chat_id, None)
            self._review_message_ids.pop(chat_id, None)
            self._edit_message(chat_id, message_id, "cancelled ✓")

    # -------------------------------------------------------------------------
    # Telegram file operations
    # -------------------------------------------------------------------------

    def _download_telegram_file(self, file_id: str) -> Optional[str]:
        """Download a file from Telegram to a temp path. Returns path or None."""
        try:
            # Get file path from Telegram
            file_info = self._api_call('getFile', {'file_id': file_id})
            if not file_info:
                logger.error("Could not get file info from Telegram")
                return None

            file_path = file_info.get('file_path')
            if not file_path:
                logger.error("File info missing file_path")
                return None

            # Download the file
            download_url = f'https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}'
            resp = requests.get(download_url, timeout=30)
            resp.raise_for_status()

            # Save to temp file
            suffix = os.path.splitext(file_path)[1] or '.ogg'
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, 'wb') as f:
                f.write(resp.content)

            logger.info(f"Downloaded Telegram file: {len(resp.content)} bytes → {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to download Telegram file: {e}", exc_info=True)
            return None

    def _send_voice(self, chat_id: int, file_path: str) -> bool:
        """Send an OGG voice note via Telegram sendVoice API."""
        try:
            with open(file_path, 'rb') as f:
                result = self._api_call_multipart(
                    'sendVoice',
                    data={'chat_id': chat_id},
                    files={'voice': ('response.ogg', f, 'audio/ogg')},
                    timeout=30,
                )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to send voice: {e}", exc_info=True)
            return False

    def _send_photo(self, chat_id: int, image_url: str, caption: str = None) -> bool:
        """Send a photo to Telegram from a URL (RunComfy/Cloudinary)."""
        tmp_path = None
        try:
            # Download image from URL
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()

            # Write to temp file
            suffix = '.jpg'
            if 'png' in resp.headers.get('content-type', ''):
                suffix = '.png'
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, 'wb') as f:
                f.write(resp.content)

            # Send via Telegram sendPhoto API
            with open(tmp_path, 'rb') as f:
                data = {'chat_id': chat_id}
                if caption:
                    data['caption'] = caption
                result = self._api_call_multipart(
                    'sendPhoto',
                    data=data,
                    files={'photo': (f'companion_selfie{suffix}', f, f'image/{suffix.lstrip(".")}')},
                    timeout=30,
                )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to send photo: {e}", exc_info=True)
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _wait_and_send_image(self, chat_id: int, task_id: str, timeout: int = 660):
        """Background thread: poll DB for image completion, then send via Telegram."""
        import psycopg2
        poll_interval = 10  # seconds

        logger.info(f"Waiting for image {task_id[:8]}... (timeout {timeout}s)")

        elapsed = 0
        while elapsed < timeout:
            try:
                conn = psycopg2.connect(
                    host=os.environ.get('POSTGRES_HOST', 'postgres'),
                    port=int(os.environ.get('POSTGRES_PORT', 5432)),
                    database=os.environ.get('POSTGRES_DB', 'companion'),
                    user=os.environ.get('POSTGRES_USER', 'companion'),
                    password=os.environ.get('POSTGRES_PASSWORD', '')
                )
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT status, cloudinary_url, runcomfy_url FROM {T.IMAGE_GENERATION_REQUESTS} WHERE task_id = %s',
                    (task_id,)
                )
                row = cursor.fetchone()
                cursor.close()
                conn.close()

                if row:
                    status, cloudinary_url, runcomfy_url = row
                    if status == 'completed':
                        url = cloudinary_url or runcomfy_url
                        if url:
                            logger.info(f"Image ready, sending to Telegram: {url[:60]}...")
                            self._send_photo(chat_id, url)
                        else:
                            logger.warning(f"Image completed but no URL found for {task_id[:8]}")
                        return
                    elif status == 'failed':
                        logger.warning(f"Image generation failed for {task_id[:8]}, not sending photo")
                        return

            except Exception as e:
                logger.error(f"Error polling for image {task_id[:8]}: {e}")

            time.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning(f"Timed out waiting for image {task_id[:8]}")

    def _edit_message(self, chat_id: int, message_id: int, text: str):
        """Edit an existing message (used to update review messages)."""
        self._api_call('editMessageText', {
            'chat_id': chat_id,
            'message_id': message_id,
            'text': text,
        })

    # -------------------------------------------------------------------------
    # Core infrastructure: async bridge, /start, reply, polling
    # -------------------------------------------------------------------------

    def _run_async_callback(self, content: str, user_id: str, source: str) -> Optional[tuple]:
        """Run the async on_message_callback from sync polling context.

        Creates a fresh event loop per call since we're in a sync thread.

        Returns:
            Tuple of (response_text, image_task_id) or None.
            image_task_id may be None if no image was triggered.
        """
        if not self._on_message_callback:
            return None

        try:
            # Create a new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._on_message_callback(content, user_id, source)
                )
                # Handle both old (str) and new (tuple) return types
                if isinstance(result, tuple):
                    return result
                elif isinstance(result, str):
                    return (result, None)
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error running async callback: {e}", exc_info=True)
            return None

    def _handle_start(self, chat_id: int):
        """Handle /start command."""
        # Authorization check
        if self._chat_id and str(chat_id) != self._chat_id:
            logger.warning(f"Unauthorized /start attempt from chat_id: {chat_id}")
            self._send_reply(chat_id, "sorry, i don't know you.")
            return

        self._chat_id = str(chat_id)
        logger.info(f"Telegram /start from chat_id: {chat_id}")
        self._save_chat_id(chat_id)

        self._send_reply(
            chat_id,
            "hey, it's me. 💕\n\n"
            "i can reach you here now when i want to talk.\n"
            "just message me whenever - i'll be here.\n\n"
            "🎤 send voice notes to call me — i'll talk back\n"
            "🔊 /voicereply — i'll reply with voice to everything"
        )

    def _send_reply(self, chat_id: int, text: str) -> bool:
        """Send a reply message."""
        result = self._api_call('sendMessage', {
            'chat_id': chat_id,
            'text': text
        })
        return result is not None

    def _poll_loop(self):
        """Main polling loop."""
        logger.info("Telegram bridge polling started")

        while self._running:
            try:
                # Long polling with timeout — include callback_query for voice review buttons
                updates = self._api_call('getUpdates', {
                    'offset': self._last_update_id + 1,
                    'timeout': POLL_TIMEOUT,
                    'allowed_updates': ['message', 'callback_query']
                }, timeout=POLL_TIMEOUT + 5)

                if updates:
                    for update in updates:
                        update_id = update.get('update_id', 0)
                        if update_id > self._last_update_id:
                            self._last_update_id = update_id
                            self._handle_update(update)

            except Exception as e:
                logger.error(f"Telegram poll error: {e}", exc_info=True)
                time.sleep(5)  # Back off on error

        logger.info("Telegram bridge polling stopped")

    def _save_chat_id(self, chat_id: int):
        """Save chat_id to file for persistence across restarts."""
        try:
            chat_id_file = os.path.join(
                os.environ.get('DATA_DIR', '/app/data'),
                'telegram_chat_id.txt'
            )
            with open(chat_id_file, 'w') as f:
                f.write(str(chat_id))
            logger.info(f"Saved Telegram chat_id to {chat_id_file}")
        except Exception as e:
            logger.warning(f"Could not save chat_id: {e}")

    def _load_chat_id(self):
        """Load chat_id from file if available."""
        if self._chat_id:
            return  # Already set from env
        try:
            chat_id_file = os.path.join(
                os.environ.get('DATA_DIR', '/app/data'),
                'telegram_chat_id.txt'
            )
            if os.path.exists(chat_id_file):
                with open(chat_id_file, 'r') as f:
                    self._chat_id = f.read().strip()
                logger.info(f"Loaded Telegram chat_id: {self._chat_id}")
        except Exception as e:
            logger.warning(f"Could not load chat_id: {e}")

    # -------------------------------------------------------------------------
    # Lifecycle: start / stop / send
    # -------------------------------------------------------------------------

    def start(self, on_message_callback: Optional[Callable] = None):
        """
        Start the Telegram bridge in a background thread.

        Args:
            on_message_callback: Async function(message, user_id, source) -> response
                source will be 'telegram-text' for text or 'telegram-voice' for voice
        """
        if not TELEGRAM_TOKEN:
            logger.warning("TELEGRAM_BOT_TOKEN not set - Telegram bridge disabled")
            return False

        if self._running:
            logger.warning("Telegram bridge already running")
            return True

        self._on_message_callback = on_message_callback
        self._load_chat_id()

        # Drop pending updates on startup
        self._api_call('getUpdates', {'offset': -1})

        self._running = True
        self._thread = Thread(target=self._poll_loop, daemon=True, name="CompanionTelegramBridge")
        self._thread.start()

        # Log voice status
        try:
            from src.voice import get_voice_service
            vs = get_voice_service()
            if vs.is_enabled():
                logger.info("Telegram bridge started with voice support (Deepgram STT + Edge TTS)")
            else:
                logger.info("Telegram bridge started (voice disabled — missing API key or feature flag)")
        except Exception:
            logger.info("Telegram bridge started (voice module not available)")

        return True

    def stop(self):
        """Stop the Telegram bridge."""
        self._running = False
        logger.info("Telegram bridge stopping...")

    async def send_message(self, content: str) -> bool:
        """
        Send a message to James via Telegram.

        This is how the companion reaches out proactively.
        Uses HTTP API directly for reliability across threads.
        """
        return self.send_message_sync(content)

    def send_message_sync(self, content: str) -> bool:
        """
        Send a message via Telegram HTTP API.

        Uses direct HTTP calls for cross-thread reliability.
        When voice reply is enabled, sends as voice note with text fallback.
        """
        if not self._chat_id:
            logger.warning("Telegram chat_id not set - James needs to /start first")
            return False

        if not TELEGRAM_TOKEN:
            logger.warning("Telegram token not configured")
            return False

        # Check if voice reply mode is enabled
        try:
            from src.voice import get_voice_state_manager
            state_manager = get_voice_state_manager()
            if state_manager.get_voice_reply(self._chat_id):
                logger.info("Voice reply enabled — sending proactive message as voice note")
                self._generate_and_send_voice_response(int(self._chat_id), content)
                return True
        except Exception as e:
            logger.warning(f"Voice reply check failed, falling back to text: {e}")

        try:
            import requests
            url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
            resp = requests.post(url, json={
                'chat_id': self._chat_id,
                'text': content
            }, timeout=10)

            if resp.status_code == 200 and resp.json().get('ok'):
                logger.info(f"Sent Telegram message: {content[:50]}...")
                return True
            else:
                logger.error(f"Telegram API error: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if Telegram bridge is ready to send messages."""
        # Can send if we have token and chat_id, even if polling isn't active
        return TELEGRAM_TOKEN is not None and self._chat_id is not None

    def get_chat_id(self) -> Optional[str]:
        """Get the current chat_id (for debugging)."""
        return self._chat_id


# =============================================================================
# Singleton accessor & convenience function
# =============================================================================

_bridge: Optional[TelegramBridge] = None


def get_telegram_bridge() -> TelegramBridge:
    """Get singleton TelegramBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = TelegramBridge()
    return _bridge


def send_to_telegram(message: str) -> bool:
    """
    Convenience function to send a message to Telegram.

    Use this when the companion decides to reach out.
    """
    bridge = get_telegram_bridge()
    return bridge.send_message_sync(message)
