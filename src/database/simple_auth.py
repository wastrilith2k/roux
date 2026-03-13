"""
Simple PostgreSQL-based authentication system.

WHAT: Password hashing (PBKDF2-SHA256 with salt), session token generation,
      token verification, and a `token_required` Flask decorator for
      protecting HTTP endpoints.

WHY:  The legacy web dashboard needs basic auth. Firebase handles the primary
      chat frontend, but the admin/cost/settings routes still use this system.
      Session tokens are stored in PostgreSQL with a 30-day expiry.

HOW:  `hash_password()` uses hashlib.pbkdf2_hmac with a random salt (stored
      alongside the hash). `create_session()` generates a 32-byte hex token
      and stores it in the `sessions` table. `token_required` decorator reads
      the token from the Flask session cookie and verifies it against the DB.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import secrets
import os
from functools import wraps
from datetime import datetime, timedelta
from contextlib import contextmanager

@contextmanager
def get_db():
    """Get PostgreSQL database connection"""
    pg_host = os.getenv('POSTGRES_HOST', 'localhost')
    pg_port = os.getenv('POSTGRES_PORT', '5432')
    pg_db = os.getenv('POSTGRES_DB', 'companion')
    pg_user = os.getenv('POSTGRES_USER', 'companion')
    pg_pass = os.getenv('POSTGRES_PASSWORD', 'companion_secure_password_change_me')

    conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        database=pg_db,
        user=pg_user,
        password=pg_pass,
        cursor_factory=RealDictCursor
    )

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_users_db():
    """Initialize users table (already created by migration, but this ensures they exist)"""
    with get_db() as conn:
        cursor = conn.cursor()

        # Users table (already created by migration script)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')

        # Sessions table (already created by migration script)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

    print("✅ PostgreSQL auth tables initialized")

def hash_password(password: str) -> str:
    """Hash a password with salt"""
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}${pwdhash.hex()}"

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a hash"""
    try:
        salt, pwdhash = password_hash.split('$')
        computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return computed_hash.hex() == pwdhash
    except:
        return False

def create_user(email: str, password: str) -> bool:
    """Create a new user"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            password_hash = hash_password(password)
            cursor.execute('INSERT INTO users (email, password_hash) VALUES (%s, %s)', (email, password_hash))
        return True
    except psycopg2.IntegrityError:
        return False  # User already exists

def authenticate_user(email: str, password: str) -> dict:
    """Authenticate a user and create a session"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        if not user or not verify_password(password, user['password_hash']):
            return None

        # Update last login
        cursor.execute('UPDATE users SET last_login = %s WHERE id = %s', (datetime.now(), user['id']))

        # Create session token
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)

        cursor.execute('''
            INSERT INTO sessions (user_id, session_token, expires_at)
            VALUES (%s, %s, %s)
        ''', (user['id'], session_token, expires_at))

        return {
            'user_id': user['id'],
            'email': user['email'],
            'session_token': session_token
        }

def verify_session(session_token: str) -> dict:
    """Verify a session token and return user info"""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute('''
            SELECT u.id, u.email, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_token = %s
        ''', (session_token,))

        result = cursor.fetchone()

        if not result:
            return None

        # Check if session expired
        expires_at = result['expires_at']
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        if datetime.now() > expires_at:
            return None

        return {
            'user_id': result['id'],
            'email': result['email']
        }

def logout_user(session_token: str):
    """Delete a session (logout)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE session_token = %s', (session_token,))

def change_password(email: str, old_password: str, new_password: str) -> bool:
    """Change a user's password"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()

        if not user:
            return False

        # Verify old password
        if not verify_password(old_password, user['password_hash']):
            return False

        # Update to new password
        new_hash = hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = %s WHERE email = %s', (new_hash, email))

    return True

def token_required(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Import Flask here to avoid eventlet monkey-patching issues
        from flask import request, jsonify

        # Get token from Authorization header
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401

        token = auth_header.replace('Bearer ', '').strip()

        # Verify session
        user_info = verify_session(token)

        if not user_info:
            return jsonify({'error': 'Invalid or expired session'}), 401

        # Add user info to request
        request.user_id = user_info['user_id']
        request.user_email = user_info['email']

        return f(*args, **kwargs)

    return decorated

# Note: Call init_users_db() manually after Flask app is created
