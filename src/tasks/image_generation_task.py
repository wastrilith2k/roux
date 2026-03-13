"""
Image Generation Task - Async image generation via RunComfy + Cloudinary.

WHAT: Submits image generation prompts to RunComfy API (ComfyUI-based service),
      polls for completion, backs up the result to Cloudinary for persistent
      hosting, notifies the frontend via Redis pub/sub, and tracks requests
      in PostgreSQL. Also includes an orphan recovery task for requests where
      the worker died mid-generation.

WHEN: On-demand -- triggered when the companion or user requests an image.
      Orphan recovery runs periodically.

WHY:  Image generation takes 30-300 seconds (plus cold start), far too long
      for synchronous response. This task handles the full async lifecycle:
      submit -> poll -> download -> backup -> notify. The Cloudinary backup
      ensures images survive RunComfy's ephemeral storage. Orphan recovery
      prevents lost images when workers crash mid-generation.
"""
import os
import logging
import redis
import json
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=660,  # 11 min soft limit (covers cold start)
    time_limit=720,  # 12 min hard limit
)
def generate_image_task(
    self,
    task_id: str,
    email: str,
    prompt: str,
    workflow_type: str = 'personal',
    width: int = 1024,
    height: int = 1024,
    room_id: str = None,
    intimate: bool = False
):
    """
    Generate an image via RunComfy and deliver it.

    Args:
        task_id: Unique task identifier
        email: User's email (for notifications)
        prompt: Image generation prompt
        workflow_type: 'personal' (with LoRA) or 'general'
        width: Image width
        height: Image height
        room_id: WebSocket room for notifications
    """
    logger.info(f"🎨 Starting image generation task: {task_id[:8]}...")
    logger.info(f"   Prompt: {prompt[:80]}...")
    logger.info(f"   Workflow: {workflow_type}, Size: {width}x{height}")

    # Save request to database for tracking
    _save_request(task_id, email, prompt, workflow_type, width, height)

    try:
        # Step 1: Generate image via RunComfy
        from src.image.runcomfy_manager import get_runcomfy_manager

        manager = get_runcomfy_manager()

        if not manager.is_enabled():
            logger.error("❌ RunComfy not enabled")
            _update_request_error(task_id, "RunComfy not enabled")
            _notify_error(room_id, task_id, "Image generation not configured")
            return {"success": False, "error": "RunComfy not enabled"}

        # Submit workflow and get request ID
        runcomfy_request_id = manager.submit_workflow(
            prompt=prompt,
            workflow_type=workflow_type,
            width=width,
            height=height,
            intimate=intimate
        )

        if not runcomfy_request_id:
            _update_request_error(task_id, "Failed to submit workflow")
            _notify_error(room_id, task_id, "Failed to submit workflow")
            return {"success": False, "error": "Failed to submit workflow"}

        # Save RunComfy request ID for later retrieval
        _update_request_runcomfy_id(task_id, runcomfy_request_id)
        logger.info(f"   RunComfy request ID: {runcomfy_request_id}")

        # --- Poll RunComfy for completion ---
        # Personal workflow uses LoRA-tuned model; general uses base model
        deployment_id = (
            manager.personal_deployment_id if workflow_type == "personal"
            else manager.general_deployment_id
        )

        import time
        start_time = time.time()
        timeout = 600   # 10 min timeout (covers cold starts + generation)
        poll_interval = 5  # Check every 5 seconds

        while time.time() - start_time < timeout:
            status_data = manager.check_status(runcomfy_request_id, deployment_id)
            if status_data:
                status = status_data.get('status')
                progress = status_data.get('progress', 0)

                if progress > 0:
                    logger.info(f"   Progress: {progress}%")

                if status in ('succeeded', 'completed'):
                    runcomfy_url = manager.get_result(runcomfy_request_id, deployment_id)
                    break
                elif status == 'failed':
                    _update_request_error(task_id, "RunComfy workflow failed")
                    _notify_error(room_id, task_id, "Image generation failed")
                    return {"success": False, "error": "Workflow failed"}

            time.sleep(poll_interval)
        else:
            _update_request_error(task_id, f"Timed out after {timeout}s")
            _notify_error(room_id, task_id, "Image generation timed out")
            return {"success": False, "error": "Timeout"}

        if not runcomfy_url:
            _update_request_error(task_id, "No image URL returned")
            _notify_error(room_id, task_id, "Image generation failed")
            return {"success": False, "error": "No image URL"}

        # --- Backup to Cloudinary for persistent hosting ---
        # RunComfy URLs are ephemeral; Cloudinary URLs persist
        cloudinary_url = manager.backup_to_cloudinary(runcomfy_url, task_id)

        _update_request_success(task_id, runcomfy_url, cloudinary_url)

        # Prefer Cloudinary URL (persistent) over RunComfy URL (ephemeral)
        final_url = cloudinary_url or runcomfy_url
        logger.info(f"✅ Image generated: {final_url}")

        # Step 2: Notify via Redis pub/sub (for WebSocket)
        _notify_success(room_id, task_id, final_url, prompt)

        return {
            "success": True,
            "url": final_url,
            "cloudinary_url": cloudinary_url,
            "runcomfy_url": runcomfy_url,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"❌ Image generation error: {e}", exc_info=True)
        _notify_error(room_id, task_id, str(e))

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        return {"success": False, "error": str(e)}


def _get_redis():
    """Get Redis client for pub/sub."""
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'redis'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=True
    )


def _notify_success(room_id: str, task_id: str, url: str, prompt: str):
    """Notify via Redis pub/sub that image is ready."""
    if not room_id:
        return

    try:
        r = _get_redis()
        message = {
            "type": "image_generated",
            "task_id": task_id,
            "url": url,
            "prompt": prompt,
            "room_id": room_id
        }
        r.publish("image_generated", json.dumps(message))
        logger.info(f"📤 Published image_generated event to Redis")
    except Exception as e:
        logger.warning(f"Failed to publish to Redis: {e}")


def _notify_error(room_id: str, task_id: str, error: str):
    """Notify via Redis pub/sub that image generation failed."""
    if not room_id:
        return

    try:
        r = _get_redis()
        message = {
            "type": "image_error",
            "task_id": task_id,
            "error": error,
            "room_id": room_id
        }
        r.publish("image_generated", json.dumps(message))
    except Exception as e:
        logger.warning(f"Failed to publish error to Redis: {e}")


# Database tracking functions
def _get_db_connection():
    """Get PostgreSQL connection."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'companion'),
        user=os.getenv('POSTGRES_USER', 'companion'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )


def _save_request(task_id: str, email: str, prompt: str, workflow_type: str, width: int, height: int):
    """Save image generation request to database."""
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO image_generation_requests
            (task_id, email, prompt, workflow_type, width, height, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            ON CONFLICT (task_id) DO NOTHING
        ''', (task_id, email, prompt, workflow_type, width, height))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"💾 Saved request to database: {task_id[:8]}...")
    except Exception as e:
        logger.warning(f"Failed to save request to database: {e}")


def _update_request_runcomfy_id(task_id: str, runcomfy_request_id: str):
    """Update request with RunComfy request ID."""
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE image_generation_requests
            SET runcomfy_request_id = %s, status = 'processing'
            WHERE task_id = %s
        ''', (runcomfy_request_id, task_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to update RunComfy ID: {e}")


def _update_request_success(task_id: str, runcomfy_url: str, cloudinary_url: str = None):
    """Update request with successful generation URLs."""
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE image_generation_requests
            SET runcomfy_url = %s, cloudinary_url = %s, status = 'completed',
                downloaded_at = CURRENT_TIMESTAMP,
                uploaded_at = CASE WHEN %s IS NOT NULL THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE task_id = %s
        ''', (runcomfy_url, cloudinary_url, cloudinary_url, task_id))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"💾 Updated request with URLs: {task_id[:8]}...")
    except Exception as e:
        logger.warning(f"Failed to update request URLs: {e}")


def _update_request_error(task_id: str, error_message: str):
    """Update request with error status."""
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE image_generation_requests
            SET status = 'failed', error_message = %s
            WHERE task_id = %s
        ''', (error_message, task_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to update request error: {e}")


@shared_task(
    bind=True,
    max_retries=0,
    soft_time_limit=120,
    time_limit=180,
)
def recover_orphaned_images(self):
    """
    Recover orphaned image generation requests.

    Finds requests stuck in 'pending' or 'processing' status where the
    celery worker died before completing. Checks RunComfy for results
    and either recovers completed images or marks them as failed.
    """
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()

        # Find orphaned requests: processing for > 2 minutes
        # (normal generation takes 30-300s, so 2 min pending = orphaned)
        cursor.execute('''
            SELECT task_id, email, runcomfy_request_id, workflow_type, status,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - requested_at)) as age_seconds
            FROM image_generation_requests
            WHERE status IN ('pending', 'processing')
            AND requested_at < CURRENT_TIMESTAMP - INTERVAL '2 minutes'
            ORDER BY requested_at DESC
            LIMIT 10
        ''')
        orphans = cursor.fetchall()
        cursor.close()
        conn.close()

        if not orphans:
            return {"recovered": 0, "failed": 0, "message": "No orphans found"}

        logger.info(f"Found {len(orphans)} orphaned image request(s)")

        from src.image.runcomfy_manager import get_runcomfy_manager
        manager = get_runcomfy_manager()

        recovered = 0
        failed = 0

        for task_id, email, runcomfy_request_id, workflow_type, status, age_seconds in orphans:
            logger.info(f"Checking orphan {task_id[:8]}... (age: {age_seconds:.0f}s, status: {status})")

            # If no RunComfy request ID, it never made it to RunComfy
            if not runcomfy_request_id:
                if age_seconds > 300:  # 5 min without RunComfy ID = dead
                    _update_request_error(task_id, "Orphaned: never submitted to RunComfy")
                    failed += 1
                    logger.info(f"  Marked {task_id[:8]} as failed (never submitted)")
                continue

            # Check RunComfy status
            deployment_id = (
                manager.personal_deployment_id if workflow_type == 'personal'
                else manager.general_deployment_id
            )

            try:
                status_data = manager.check_status(runcomfy_request_id, deployment_id)

                if not status_data:
                    # API unreachable — skip, try again next cycle
                    if age_seconds > 900:  # 15 min = give up
                        _update_request_error(task_id, "Orphaned: RunComfy API unreachable for too long")
                        failed += 1
                    continue

                rc_status = status_data.get('status')

                if rc_status in ('succeeded', 'completed'):
                    # Image is ready — fetch and recover
                    runcomfy_url = manager.get_result(runcomfy_request_id, deployment_id)
                    if runcomfy_url:
                        cloudinary_url = manager.backup_to_cloudinary(runcomfy_url, task_id)
                        _update_request_success(task_id, runcomfy_url, cloudinary_url)

                        final_url = cloudinary_url or runcomfy_url
                        _notify_success(email, task_id, final_url, "recovered orphaned image")

                        # Try to send to Telegram
                        _send_recovered_to_telegram(final_url)

                        recovered += 1
                        logger.info(f"  Recovered {task_id[:8]}: {final_url[:60]}...")
                    else:
                        _update_request_error(task_id, "Orphaned: completed but no URL returned")
                        failed += 1

                elif rc_status == 'failed':
                    _update_request_error(task_id, "Orphaned: RunComfy workflow failed")
                    failed += 1

                else:
                    # Still processing on RunComfy — leave it, check again next cycle
                    if age_seconds > 900:  # 15 min = too long
                        _update_request_error(task_id, f"Orphaned: timed out after {age_seconds:.0f}s")
                        failed += 1
                    else:
                        logger.info(f"  {task_id[:8]} still processing on RunComfy, will check again")

            except Exception as e:
                logger.error(f"  Error checking orphan {task_id[:8]}: {e}")

        result = {"recovered": recovered, "failed": failed, "total_orphans": len(orphans)}
        logger.info(f"Orphan recovery complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Orphan recovery task error: {e}", exc_info=True)
        return {"error": str(e)}


def _send_recovered_to_telegram(image_url: str):
    """Send a recovered image to Telegram directly via API."""
    import tempfile

    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        return

    # Try to get chat_id from file (shared volume)
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not chat_id:
        chat_id_file = os.path.join(os.getenv('DATA_DIR', '/app/data'), 'telegram_chat_id.txt')
        try:
            with open(chat_id_file, 'r') as f:
                chat_id = f.read().strip()
        except Exception:
            logger.debug("No Telegram chat_id available for image recovery")
            return

    if not chat_id:
        return

    tmp_path = None
    try:
        # Download image
        import requests as req
        resp = req.get(image_url, timeout=30)
        resp.raise_for_status()

        suffix = '.png' if 'png' in resp.headers.get('content-type', '') else '.jpg'
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            f.write(resp.content)

        # Send via Telegram API
        with open(tmp_path, 'rb') as f:
            send_resp = req.post(
                f'https://api.telegram.org/bot{token}/sendPhoto',
                data={'chat_id': chat_id},
                files={'photo': (f'companion_recovered{suffix}', f, f'image/{suffix.lstrip(".")}')},
                timeout=30,
            )
        if send_resp.json().get('ok'):
            logger.info(f"Sent recovered image to Telegram")
        else:
            logger.warning(f"Telegram sendPhoto failed: {send_resp.json().get('description')}")
    except Exception as e:
        logger.debug(f"Could not send recovered image to Telegram: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
