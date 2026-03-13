"""Tests for Firebase authentication security."""

import pytest
from unittest.mock import patch, MagicMock
from src.auth.firebase_auth import verify_firebase_token


class TestVerifyFirebaseToken:
    """Test Firebase token verification security."""

    def test_empty_token_returns_none(self):
        assert verify_firebase_token("") is None
        assert verify_firebase_token(None) is None

    @patch('src.auth.firebase_auth.get_firebase_public_keys')
    @patch('src.auth.firebase_auth.jwt')
    def test_rejects_when_keys_unavailable(self, mock_jwt, mock_keys):
        """When Firebase public keys can't be fetched, tokens MUST be rejected."""
        mock_jwt.decode.return_value = {
            'aud': 'my-project',
            'iss': 'https://securetoken.google.com/my-project',
            'email': 'user@test.com',
        }
        mock_keys.return_value = {}  # Keys unavailable

        result = verify_firebase_token("fake.jwt.token")
        assert result is None  # Must reject, not fall back to unverified

    @patch('src.auth.firebase_auth.get_firebase_public_keys')
    @patch('src.auth.firebase_auth.jwt')
    def test_rejects_wrong_issuer(self, mock_jwt, mock_keys):
        """Tokens with wrong issuer should be rejected."""
        mock_jwt.decode.return_value = {
            'aud': 'my-project',
            'iss': 'https://evil.com/my-project',
            'email': 'user@test.com',
        }

        result = verify_firebase_token("fake.jwt.token")
        assert result is None

    @patch('src.auth.firebase_auth.get_firebase_public_keys')
    @patch('src.auth.firebase_auth.jwt')
    def test_rejects_missing_audience(self, mock_jwt, mock_keys):
        """Tokens without audience claim should be rejected."""
        mock_jwt.decode.return_value = {
            'iss': 'https://securetoken.google.com/my-project',
            'email': 'user@test.com',
        }

        result = verify_firebase_token("fake.jwt.token")
        assert result is None
