#!/usr/bin/env python3
"""
One-time OAuth consent flow to get a refresh token for Google API access.

Usage:
    python3 scripts/google_oauth_setup.py                    # the companion's full token
    python3 scripts/google_oauth_setup.py --user-calendar    # user's calendar-only token

This will:
1. Print a URL — open it in your browser
2. Log into the Google account and authorize the app
3. You'll be redirected to localhost (which will fail) — copy the full URL
4. Paste it back here
5. The script saves the refresh token to credentials/
"""

import json
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

# All scopes for the companion's full access
COMPANION_SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/chat.spaces',
    'https://www.googleapis.com/auth/chat.spaces.readonly',
    'https://www.googleapis.com/auth/chat.messages.create',
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/photoslibrary.appendonly',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/presentations',
    'https://www.googleapis.com/auth/tasks',
]

# Minimal scopes for user calendar reads
USER_CALENDAR_SCOPES = [
    'https://www.googleapis.com/auth/calendar.readonly',
]

CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), '..', 'credentials', 'google_oauth_client.json')
COMPANION_TOKEN_FILE = os.path.join(os.path.dirname(__file__), '..', 'credentials', 'google_token.json')
USER_CALENDAR_TOKEN_FILE = os.path.join(os.path.dirname(__file__), '..', 'credentials', 'user_calendar_token.json')


def run_flow(scopes, token_file, label):
    if not os.path.exists(CLIENT_SECRET_FILE):
        # Fall back to inline client config
        client_config = {
            "installed": {
                "client_id": os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "YOUR_CLIENT_ID"),
                "client_secret": os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
        flow = InstalledAppFlow.from_client_config(
            client_config, scopes=scopes, redirect_uri='http://localhost'
        )
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE, scopes=scopes, redirect_uri='http://localhost'
        )

    if os.path.exists(token_file):
        print(f"Token file already exists at {token_file}")
        resp = input("Overwrite? (y/N): ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            sys.exit(0)

    print(f"\n=== {label} OAuth Setup ===\n")
    print("Requesting scopes:")
    for scope in scopes:
        short = scope.split('/')[-1]
        print(f"  - {short}")
    print()

    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
    )

    print("1. Open this URL in your browser:\n")
    print(f"   {auth_url}\n")
    print("2. Log in and authorize the app.")
    print("3. You'll be redirected to http://localhost/?code=... (page won't load, that's OK)")
    print("4. Copy the FULL URL from your browser's address bar and paste it below.\n")

    redirect_response = input("Paste the full redirect URL here: ").strip()

    if not redirect_response:
        print("No URL provided. Aborted.")
        sys.exit(1)

    flow.fetch_token(authorization_response=redirect_response)
    creds = flow.credentials

    token_data = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': list(creds.scopes) if creds.scopes else scopes,
    }

    os.makedirs(os.path.dirname(token_file), exist_ok=True)
    with open(token_file, 'w') as f:
        json.dump(token_data, f, indent=2)

    print(f"\nToken saved to {token_file}")
    print(f"Refresh token: {creds.refresh_token[:20]}...")


def main():
    if '--user-calendar' in sys.argv:
        print("Setting up calendar-only access for your personal Google account.")
        print("Log in as YOURSELF (not the companion) when the browser opens.\n")
        run_flow(USER_CALENDAR_SCOPES, USER_CALENDAR_TOKEN_FILE, "User Calendar")
    else:
        print("Setting up full access for the companion's Google account.")
        print("Log in as THE COMPANION when the browser opens.\n")
        run_flow(COMPANION_SCOPES, COMPANION_TOKEN_FILE, "Companion Google API")


if __name__ == '__main__':
    main()
