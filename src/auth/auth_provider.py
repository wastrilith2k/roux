"""
Authentication Provider — abstraction layer for identity providers.

WHAT: Defines a common interface for verifying ID tokens and a factory
      that returns the configured provider. Currently supports Firebase
      and AWS Cognito.

WHY:  Decouples token verification from the specific identity service.
      Set AUTH_PROVIDER=firebase or AUTH_PROVIDER=cognito in your
      environment to switch providers without code changes. All callers
      use verify_token() / token_required() from this module.

HOW:  AuthProvider is an abstract base class. get_auth_provider() returns
      a singleton of the configured concrete class. The module also
      exports convenience functions (verify_token, auth_token_required,
      auth_token_optional) that delegate to the active provider.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    def initialize(self) -> bool:
        """Perform any one-time setup. Called at app startup."""
        ...

    @abstractmethod
    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify an ID token and return decoded claims.

        Returns a dict with at least:
            - uid: str   (provider-specific user ID)
            - email: str
            - email_verified: bool
        Or None if the token is invalid.
        """
        ...

    @abstractmethod
    def get_user_info(self, uid: str) -> Optional[Dict[str, Any]]:
        """Return user info for a UID. May just echo back token claims."""
        ...


# ---------------------------------------------------------------------------
# Firebase implementation
# ---------------------------------------------------------------------------

class FirebaseAuthProvider(AuthProvider):
    """Firebase ID token verification (lightweight JWT, no firebase-admin)."""

    def initialize(self) -> bool:
        from src.auth.firebase_auth import initialize_firebase
        return initialize_firebase()

    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        from src.auth.firebase_auth import verify_firebase_token
        return verify_firebase_token(id_token)

    def get_user_info(self, uid: str) -> Optional[Dict[str, Any]]:
        from src.auth.firebase_auth import get_firebase_user_info
        return get_firebase_user_info(uid)


# ---------------------------------------------------------------------------
# AWS Cognito implementation
# ---------------------------------------------------------------------------

class CognitoAuthProvider(AuthProvider):
    """AWS Cognito ID token verification via JWKS."""

    def __init__(self):
        self.user_pool_id = os.environ.get('AWS_COGNITO_USER_POOL_ID', '')
        self.client_id = os.environ.get('AWS_COGNITO_CLIENT_ID', '')
        self.region = os.environ.get('AWS_COGNITO_REGION', os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'))
        self._jwks_cache = None

    def _get_issuer(self) -> str:
        return f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"

    def _get_jwks_url(self) -> str:
        return f"{self._get_issuer()}/.well-known/jwks.json"

    def _get_jwks(self) -> dict:
        if self._jwks_cache:
            return self._jwks_cache
        try:
            import requests
            resp = requests.get(self._get_jwks_url(), timeout=5)
            resp.raise_for_status()
            self._jwks_cache = resp.json()
            logger.info("Fetched Cognito JWKS for token verification")
            return self._jwks_cache
        except Exception as e:
            logger.warning(f"Could not fetch Cognito JWKS: {e}")
            return {}

    def initialize(self) -> bool:
        if not self.user_pool_id or not self.client_id:
            logger.warning("AWS Cognito not fully configured (missing USER_POOL_ID or CLIENT_ID)")
            return False
        logger.info(f"Cognito auth initialized (pool={self.user_pool_id}, region={self.region})")
        return True

    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        if not id_token:
            return None

        try:
            import jwt
            from jwt import PyJWKClient

            jwks_url = self._get_jwks_url()
            jwk_client = PyJWKClient(jwks_url)
            signing_key = jwk_client.get_signing_key_from_jwt(id_token)

            decoded = jwt.decode(
                id_token,
                signing_key.key,
                algorithms=['RS256'],
                audience=self.client_id,
                issuer=self._get_issuer(),
                options={"leeway": 120},
            )

            # Cognito ID tokens use 'sub' for user id and 'email' for email
            # Normalize to match the interface expected by the rest of the app
            return {
                'uid': decoded.get('sub'),
                'email': decoded.get('email'),
                'email_verified': decoded.get('email_verified', False),
                'iss': decoded.get('iss'),
                'aud': decoded.get('aud'),
                'cognito:username': decoded.get('cognito:username'),
            }

        except Exception as e:
            logger.warning(f"Cognito token verification failed: {e}")
            return None

    def get_user_info(self, uid: str) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(request, 'auth_user') and request.auth_user:
                claims = request.auth_user
                return {
                    'uid': claims.get('uid'),
                    'email': claims.get('email'),
                    'email_verified': claims.get('email_verified', False),
                }
        except RuntimeError:
            pass
        return {'uid': uid, 'email': None, 'email_verified': False}


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

_provider: Optional[AuthProvider] = None


def get_auth_provider() -> AuthProvider:
    """Return the configured auth provider singleton.

    Reads AUTH_PROVIDER env var:
        'firebase' (default) — Firebase ID token verification
        'cognito'            — AWS Cognito ID token verification
    """
    global _provider
    if _provider is None:
        choice = os.environ.get('AUTH_PROVIDER', 'firebase').lower()
        if choice == 'cognito':
            _provider = CognitoAuthProvider()
        else:
            _provider = FirebaseAuthProvider()
        _provider.initialize()
        logger.info(f"Auth provider: {choice}")
    return _provider


# ---------------------------------------------------------------------------
# Convenience functions — used by routes and middleware
# ---------------------------------------------------------------------------

def verify_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verify an ID token using the active auth provider."""
    return get_auth_provider().verify_token(id_token)


def get_user_info(uid: str) -> Optional[Dict[str, Any]]:
    """Get user info using the active auth provider."""
    return get_auth_provider().get_user_info(uid)


def auth_token_required(f):
    """Decorator requiring a valid ID token (any configured provider).

    Sets on request:
        request.auth_user  — full decoded claims
        request.user_id    — provider UID
        request.user_email — user email
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        token = auth_header[7:]
        claims = verify_token(token)
        if not claims:
            return jsonify({'error': 'Invalid or expired token'}), 401

        request.auth_user = claims
        request.user_id = claims.get('uid')
        request.user_email = claims.get('email')
        # Backwards compat for code using the firebase-specific attribute
        request.firebase_user = claims

        return f(*args, **kwargs)
    return decorated


def auth_token_optional(f):
    """Decorator allowing optional ID token authentication.

    Same attributes as auth_token_required but set to None when unauthenticated.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            claims = verify_token(token)
            if claims:
                request.auth_user = claims
                request.user_id = claims.get('uid')
                request.user_email = claims.get('email')
                request.firebase_user = claims
            else:
                request.auth_user = None
                request.user_id = None
                request.user_email = None
                request.firebase_user = None
        else:
            request.auth_user = None
            request.user_id = None
            request.user_email = None
            request.firebase_user = None

        return f(*args, **kwargs)
    return decorated
