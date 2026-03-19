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

    def backup_to_cloudinary(self, image_url, task_id):
        try:
            import cloudinary
            import cloudinary.uploader
            cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
            api_key = os.environ.get('CLOUDINARY_API_KEY')
            api_secret = os.environ.get('CLOUDINARY_API_SECRET')
            if not all([cloud_name, api_key, api_secret]):
                return None
            cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret)
            result = cloudinary.uploader.upload(image_url, public_id=f"companion/{task_id}")
            return result.get('secure_url')
        except Exception as e:
            logger.warning(f"Cloudinary backup failed: {e}")
            return None


_manager = None

def get_runcomfy_manager() -> RunComfyManager:
    global _manager
    if _manager is None:
        _manager = RunComfyManager()
    return _manager
