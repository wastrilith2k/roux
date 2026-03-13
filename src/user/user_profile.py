"""
User profile management with display names and admin status
"""

from datetime import datetime
from typing import Optional, Dict

class UserProfile:
    """Manage user profiles with display names and admin status"""

    def __init__(self, chroma_client):
        self.client = chroma_client
        self.collection = self.client.get_or_create_collection('user_profiles')

    def get_profile(self, email: str) -> Dict:
        """Get user profile by email"""
        profile_id = f"profile_{email}"

        try:
            results = self.collection.get(ids=[profile_id])
            if results['ids']:
                metadata = results['metadatas'][0]
                return {
                    'email': email,
                    'display_name': metadata.get('display_name', email.split('@')[0]),
                    'is_admin': metadata.get('is_admin', False),
                    'created_at': metadata.get('created_at', ''),
                    'last_seen': metadata.get('last_seen', '')
                }
        except:
            pass

        # Return default profile if not found
        return {
            'email': email,
            'display_name': email.split('@')[0],  # Use part before @ as default
            'is_admin': False,
            'created_at': '',
            'last_seen': ''
        }

    def set_display_name(self, email: str, display_name: str):
        """Set or update user's display name"""
        profile_id = f"profile_{email}"
        existing = self.get_profile(email)

        metadata = {
            'email': email,
            'display_name': display_name,
            'is_admin': existing['is_admin'],
            'created_at': existing['created_at'] or datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat()
        }

        try:
            # Try to update
            self.collection.update(
                ids=[profile_id],
                metadatas=[metadata]
            )
        except:
            # If not exists, create
            self.collection.add(
                ids=[profile_id],
                documents=[f"User profile for {email}"],
                metadatas=[metadata]
            )

    def set_admin(self, email: str, is_admin: bool = True):
        """Set user as admin (can modify companion personality)"""
        profile_id = f"profile_{email}"
        existing = self.get_profile(email)

        metadata = {
            'email': email,
            'display_name': existing['display_name'],
            'is_admin': is_admin,
            'created_at': existing['created_at'] or datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat()
        }

        try:
            self.collection.update(
                ids=[profile_id],
                metadatas=[metadata]
            )
        except:
            self.collection.add(
                ids=[profile_id],
                documents=[f"User profile for {email}"],
                metadatas=[metadata]
            )

    def is_admin(self, email: str) -> bool:
        """Check if user has admin privileges"""
        profile = self.get_profile(email)
        return profile['is_admin']

    def update_last_seen(self, email: str):
        """Update last seen timestamp"""
        profile_id = f"profile_{email}"
        existing = self.get_profile(email)

        metadata = {
            'email': email,
            'display_name': existing['display_name'],
            'is_admin': existing['is_admin'],
            'created_at': existing['created_at'] or datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat()
        }

        try:
            self.collection.update(
                ids=[profile_id],
                metadatas=[metadata]
            )
        except:
            self.collection.add(
                ids=[profile_id],
                documents=[f"User profile for {email}"],
                metadatas=[metadata]
            )
