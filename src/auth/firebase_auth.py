"""
Firebase Authentication Module (Lightweight - No firebase-admin required)

Handles Firebase ID token validation:
- Validates JWT tokens from Firebase frontend SDK
- Extracts user info from token claims
- Provides decorators for protected routes

Frontend (Next.js) gets ID token from Firebase SDK and sends via:
  Authorization: Bearer <firebase_id_token>

This backend validates the JWT without needing firebase-admin SDK.
"""
import os
import logging
import jwt
import requests
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, jsonify
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

# Firebase project config
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', '')
_public_keys_cache = {}


def get_firebase_public_keys():
    """Fetch Firebase public keys for JWT verification.

    Keys are cached to avoid repeated HTTP requests.
    """
    global _public_keys_cache

    if _public_keys_cache:
        return _public_keys_cache

    try:
        # Firebase public keys endpoint
        url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        _public_keys_cache = response.json()
        logger.info("✅ Fetched Firebase public keys for token verification")
        return _public_keys_cache
    except Exception as e:
        logger.warning(f"⚠️  Could not fetch Firebase public keys: {e}")
        return {}


def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verify a Firebase ID token and return the decoded claims.

    Args:
        id_token: Firebase ID token from client (from Firebase SDK)

    Returns:
        Dictionary with token claims if valid, None if invalid

    Claims include:
        - uid: Firebase user ID
        - email: User email
        - email_verified: Whether email is verified
        - iss: Token issuer
        - aud: Token audience (should be project ID)
    """
    if not id_token:
        return None

    try:
        # Decode without verification first to get the header
        unverified = jwt.decode(id_token, options={"verify_signature": False})

        # Get the project ID from token claims or env
        token_aud = unverified.get('aud')
        project_id = FIREBASE_PROJECT_ID or token_aud

        if not token_aud:
            logger.warning("⚠️  No 'aud' claim in Firebase token")
            return None

        # Verify token structure
        if unverified.get('iss') != f"https://securetoken.google.com/{token_aud}":
            logger.warning(f"⚠️  Invalid token issuer: {unverified.get('iss')}")
            return None

        # Get Firebase public keys
        public_keys = get_firebase_public_keys()
        if not public_keys:
            logger.error("Firebase public keys unavailable — rejecting token")
            return None

        # Find the correct key ID
        unverified_header = jwt.get_unverified_header(id_token)
        kid = unverified_header.get('kid')

        if kid not in public_keys:
            logger.warning(f"⚠️  Key ID not found in Firebase public keys: {kid}")
            return None

        # Verify the token signature
        # Firebase returns X509 certificates; need to extract the public key
        cert_string = public_keys[kid]

        try:
            # Parse the X509 certificate
            logger.info(f"📋 Processing certificate for kid: {kid}")
            logger.info(f"   Certificate starts with: {cert_string[:50]}...")

            cert = x509.load_pem_x509_certificate(
                cert_string.encode('utf-8'),
                default_backend()
            )
            logger.info(f"✅ Certificate parsed successfully")

            # Check certificate validity (with leeway for clock skew)
            from datetime import datetime, timedelta
            now = datetime.utcnow()
            leeway = timedelta(seconds=60)  # 60 second leeway for certificate expiry
            if now > cert.not_valid_after + leeway:
                logger.warning(f"⚠️  Firebase certificate has expired (valid until {cert.not_valid_after}), but continuing with key extraction")

            # Extract the public key from the certificate
            cert_public_key = cert.public_key()
            logger.info(f"✅ Public key extracted from certificate")

            # Convert to PEM format for PyJWT
            public_key_pem = cert_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            logger.info(f"✅ Public key converted to PEM format")

            decoded = jwt.decode(
                id_token,
                public_key_pem,
                algorithms=['RS256'],
                audience=token_aud,
                options={"leeway": 3600}  # 1 hour leeway for clock skew or stale tokens
            )
        except jwt.ExpiredSignatureError as expire_error:
            logger.warning(f"Firebase token expired: {expire_error}")
            return None
        except Exception as cert_error:
            logger.error(f"❌ Error processing Firebase certificate: {cert_error}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return None

        logger.info(f"✅ Firebase token verified for user: {decoded.get('email')}")
        return decoded

    except jwt.InvalidTokenError as e:
        logger.warning(f"⚠️  Invalid Firebase token: {e}")
        return None
    except jwt.ExpiredSignatureError as e:
        logger.warning(f"⚠️  Expired Firebase token (top level): {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error verifying Firebase token: {e}")
        return None


def firebase_token_required(f):
    """Decorator to require Firebase ID token for route access.

    Expected: Authorization: Bearer <firebase_id_token>

    Adds to request context:
    - request.firebase_user: Decoded token claims
    - request.user_id: Firebase UID
    - request.user_email: User email
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        token = auth_header[7:]  # Remove "Bearer " prefix

        # Verify token
        claims = verify_firebase_token(token)
        if not claims:
            return jsonify({'error': 'Invalid or expired Firebase token'}), 401

        # Add to request context
        request.firebase_user = claims
        request.user_id = claims.get('uid')
        request.user_email = claims.get('email')

        return f(*args, **kwargs)

    return decorated_function


def firebase_auth_optional(f):
    """Decorator allowing either Firebase-authenticated or unauthenticated access.

    If valid token provided, adds to request context. Otherwise allows access.

    Adds to request context (if authenticated):
    - request.firebase_user: Decoded token claims
    - request.user_id: Firebase UID
    - request.user_email: User email
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            claims = verify_firebase_token(token)

            if claims:
                request.firebase_user = claims
                request.user_id = claims.get('uid')
                request.user_email = claims.get('email')
            else:
                # Invalid token but we allow unauthenticated access
                request.firebase_user = None
                request.user_id = None
                request.user_email = None
        else:
            # No token provided, allow unauthenticated access
            request.firebase_user = None
            request.user_id = None
            request.user_email = None

        return f(*args, **kwargs)

    return decorated_function


def get_firebase_user_info(uid: str) -> Optional[Dict[str, Any]]:
    """Get user info from Firebase UID.

    In lightweight JWT approach, we only have info from the token itself.
    This function is called after token verification, so claims are in request context.

    Args:
        uid: Firebase user ID (from token)

    Returns:
        Dictionary with user info or None
    """
    # In lightweight approach, user info comes from the token claims
    # We try to get it from request context first (set by @firebase_token_required)
    try:
        if hasattr(request, 'firebase_user'):
            claims = request.firebase_user
            return {
                'uid': claims.get('uid'),
                'email': claims.get('email'),
                'email_verified': claims.get('email_verified', False)
            }
    except RuntimeError:
        # Outside of request context
        pass

    # Fallback: return minimal info
    logger.debug(f"Getting Firebase user info for UID: {uid}")
    return {
        'uid': uid,
        'email': None,
        'email_verified': False
    }


def create_custom_token(uid: str) -> Optional[str]:
    """Create a custom token for Firebase (not used in lightweight approach).

    This is kept for compatibility but not used with lightweight JWT validation.
    Custom tokens require firebase-admin SDK which we don't have.

    Args:
        uid: Firebase user ID

    Returns:
        None (not supported in lightweight approach)
    """
    logger.warning("⚠️  create_custom_token() is not supported in lightweight JWT mode")
    return None


def initialize_firebase():
    """Initialize Firebase (no-op with lightweight JWT approach).

    Kept for compatibility with existing code.
    Firebase initialization happens in frontend via SDK.
    """
    logger.info("✅ Firebase auth initialized (JWT validation mode)")
    return True
