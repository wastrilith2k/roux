"""
Image Storage Provider — abstraction layer for image CDN/hosting.

WHAT: Defines a common interface for image storage backends and a factory
      that returns the configured provider. Currently supports Cloudinary
      and AWS S3.

WHY:  Decouples image upload logic from the specific hosting service.
      Set IMAGE_STORAGE_PROVIDER=cloudinary or IMAGE_STORAGE_PROVIDER=s3
      in your environment to switch providers without code changes.

HOW:  ImageStorageProvider is an abstract base class with upload() and
      get_url() methods. get_image_storage_provider() returns a singleton
      of the configured concrete class.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ImageStorageProvider(ABC):
    """Abstract base class for image storage backends."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this provider has all required credentials."""
        ...

    @abstractmethod
    def upload(self, image_url: str, key: str) -> Optional[str]:
        """Upload an image from a URL and return a persistent public URL.

        Args:
            image_url: Source URL to download the image from.
            key: Unique identifier used as the storage key / public_id.

        Returns:
            Persistent HTTPS URL for the uploaded image, or None on failure.
        """
        ...


class CloudinaryStorageProvider(ImageStorageProvider):
    """Cloudinary image storage backend."""

    def __init__(self):
        self.cloud_name = os.environ.get('CLOUDINARY_CLOUD_NAME')
        self.api_key = os.environ.get('CLOUDINARY_API_KEY')
        self.api_secret = os.environ.get('CLOUDINARY_API_SECRET')

    def is_configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)

    def upload(self, image_url: str, key: str) -> Optional[str]:
        if not self.is_configured():
            return None
        try:
            import cloudinary
            import cloudinary.uploader
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            result = cloudinary.uploader.upload(
                image_url,
                public_id=f"companion/{key}",
            )
            return result.get('secure_url')
        except Exception as e:
            logger.warning(f"Cloudinary upload failed: {e}")
            return None


class S3StorageProvider(ImageStorageProvider):
    """AWS S3 image storage backend."""

    def __init__(self):
        self.bucket = os.environ.get('AWS_S3_BUCKET')
        self.region = os.environ.get('AWS_S3_REGION', 'us-east-1')
        self.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.endpoint_url = os.environ.get('AWS_S3_ENDPOINT_URL')  # For S3-compatible services
        self.prefix = os.environ.get('AWS_S3_KEY_PREFIX', 'companion')

    def is_configured(self) -> bool:
        return bool(self.bucket and self.access_key and self.secret_key)

    def upload(self, image_url: str, key: str) -> Optional[str]:
        if not self.is_configured():
            return None
        try:
            import boto3
            import requests

            # Download image from source URL
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get('content-type', 'image/png')
            ext = 'png' if 'png' in content_type else 'jpg'
            s3_key = f"{self.prefix}/{key}.{ext}"

            # Build S3 client
            client_kwargs = {
                'service_name': 's3',
                'region_name': self.region,
                'aws_access_key_id': self.access_key,
                'aws_secret_access_key': self.secret_key,
            }
            if self.endpoint_url:
                client_kwargs['endpoint_url'] = self.endpoint_url

            s3 = boto3.client(**client_kwargs)
            s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=resp.content,
                ContentType=content_type,
                ACL='public-read',
            )

            # Build public URL
            if self.endpoint_url:
                url = f"{self.endpoint_url.rstrip('/')}/{self.bucket}/{s3_key}"
            else:
                url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

            return url
        except Exception as e:
            logger.warning(f"S3 upload failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

_provider: Optional[ImageStorageProvider] = None


def get_image_storage_provider() -> ImageStorageProvider:
    """Return the configured image storage provider singleton.

    Reads IMAGE_STORAGE_PROVIDER env var:
        'cloudinary' (default) — use Cloudinary
        's3'                   — use AWS S3
    """
    global _provider
    if _provider is None:
        choice = os.environ.get('IMAGE_STORAGE_PROVIDER', 'cloudinary').lower()
        if choice == 's3':
            _provider = S3StorageProvider()
        else:
            _provider = CloudinaryStorageProvider()
        configured = _provider.is_configured()
        logger.info(f"Image storage provider: {choice} (configured={configured})")
    return _provider
