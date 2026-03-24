"""
RunComfy Manager — image generation via RunComfy API (ComfyUI-based).

Stub module. The full implementation requires:
- RUNCOMFY_API_KEY
- RUNCOMFY_PERSONAL_DEPLOYMENT_ID and/or RUNCOMFY_GENERAL_DEPLOYMENT_ID

If these aren't configured, is_enabled() returns False and the system
falls back to NanoBanana.
"""

import os
import logging

logger = logging.getLogger(__name__)


class RunComfyManager:
    """RunComfy image generation manager."""

    def __init__(self):
        self.api_key = os.environ.get('RUNCOMFY_API_KEY')
        self.personal_deployment_id = os.environ.get('RUNCOMFY_PERSONAL_DEPLOYMENT_ID')
        self.general_deployment_id = os.environ.get('RUNCOMFY_GENERAL_DEPLOYMENT_ID')

    def is_enabled(self) -> bool:
        return bool(self.api_key and (self.personal_deployment_id or self.general_deployment_id))

    def submit_workflow(self, prompt, workflow_type='personal', width=1024, height=1024, intimate=False):
        if not self.is_enabled():
            return None
        # TODO: Implement RunComfy API call
        logger.warning("RunComfy submit_workflow not fully implemented")
        return None

    def check_status(self, request_id, deployment_id):
        return None

    def get_result(self, request_id, deployment_id):
        return None

    def backup_to_persistent_storage(self, image_url, task_id):
        """Upload an image to the configured persistent storage provider.

        Uses IMAGE_STORAGE_PROVIDER env var to select backend (cloudinary or s3).
        """
        try:
            from src.image.storage_provider import get_image_storage_provider
            provider = get_image_storage_provider()
            if not provider.is_configured():
                return None
            return provider.upload(image_url, task_id)
        except Exception as e:
            logger.warning(f"Persistent storage backup failed: {e}")
            return None

    def backup_to_cloudinary(self, image_url, task_id):
        """Backwards-compatible alias for backup_to_persistent_storage."""
        return self.backup_to_persistent_storage(image_url, task_id)


_manager = None

def get_runcomfy_manager() -> RunComfyManager:
    global _manager
    if _manager is None:
        _manager = RunComfyManager()
    return _manager
