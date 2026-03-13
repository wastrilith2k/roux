"""
Socket.IO Connection -- async client for the companion backend.

WHAT: Manages the Socket.IO connection lifecycle (connect, disconnect, reconnect)
      and maps all backend events (message, response, thinking, error, state,
      images, activity, autopilot, fact_approval_*) to a unified callback system.

WHY:  Decouples transport from UI. The CLI, and potentially other clients, can
      register callbacks and receive typed message dicts without knowing about
      Socket.IO event names or data normalization.

HOW:  Uses python-socketio AsyncClient with auto-reconnection. Each backend
      event has a dedicated handler (`_on_*`) that normalizes the data into a
      `{type, ...}` dict and emits it to all registered callbacks. Auth supports
      API key tokens or email-based fallback for local development.
"""

import os
import asyncio
import socketio
from typing import Optional, Callable
from datetime import datetime


class CompanionConnection:
    """Manages Socket.IO connection to Companion backend"""

    def __init__(self, api_key: Optional[str] = None, backend_url: str = "http://localhost:5000"):
        """
        Initialize connection

        Args:
            api_key: Authentication token
            backend_url: HTTP URL of backend
        """
        self.api_key = api_key or os.environ.get("COMPANION_API_KEY")
        self.backend_url = backend_url
        self.sio = socketio.AsyncClient(reconnection=True, reconnection_delay=1)
        self.connected = False
        self.message_callbacks = []

        # Set up event handlers
        self.sio.on('message', self._on_message)
        self.sio.on('response', self._on_response)
        self.sio.on('thinking', self._on_thinking)
        self.sio.on('error', self._on_error)
        self.sio.on('info', self._on_info)
        self.sio.on('history', self._on_history)
        self.sio.on('state', self._on_state)
        self.sio.on('images', self._on_images)
        self.sio.on('activity', self._on_activity)
        self.sio.on('autopilot', self._on_autopilot)
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)
        # Fact approval events
        self.sio.on('fact_approval_request', self._on_fact_approval_request)
        self.sio.on('pending_facts', self._on_pending_facts)
        self.sio.on('fact_approved', self._on_fact_approved)
        self.sio.on('fact_rejected', self._on_fact_rejected)
        self.sio.on('fact_edited', self._on_fact_edited)

    async def _on_connect(self):
        """Handle connection established"""
        self.connected = True

    async def _on_disconnect(self):
        """Handle disconnection"""
        self.connected = False
        await self._emit_callback({
            "type": "info",
            "message": "disconnected from backend"
        })

    async def _on_message(self, data):
        """Handle incoming message"""
        # Backend sends 'response' key, not 'content'
        if isinstance(data, str):
            content = data
        else:
            content = data.get("response") or data.get("content", "")
        await self._emit_callback({
            "type": "message",
            "content": content,
            "metadata": data if isinstance(data, dict) else None
        })

    async def _on_response(self, data):
        """Handle the companion's response"""
        await self._emit_callback({
            "type": "response",
            "content": data if isinstance(data, str) else data.get("content", ""),
            "metadata": data.get("metadata") if isinstance(data, dict) else None
        })

    async def _on_thinking(self, data):
        """Handle thinking status"""
        status = data if isinstance(data, str) else data.get("status", "thinking")
        await self._emit_callback({
            "type": "thinking",
            "status": status
        })

    async def _on_error(self, data):
        """Handle error from backend"""
        message = data if isinstance(data, str) else data.get("message", "unknown error")
        await self._emit_callback({
            "type": "error",
            "message": message
        })

    async def _on_info(self, data):
        """Handle info message"""
        message = data if isinstance(data, str) else data.get("message", "")
        await self._emit_callback({
            "type": "info",
            "message": message
        })

    async def _on_history(self, data):
        """Handle history response"""
        messages = data.get("messages", []) if isinstance(data, dict) else []
        await self._emit_callback({
            "type": "history",
            "messages": messages
        })

    async def _on_state(self, data):
        """Handle state response"""
        await self._emit_callback({
            "type": "state",
            "scene_state": data.get("scene_state") if isinstance(data, dict) else None,
            "internal_state": data.get("internal_state") if isinstance(data, dict) else None,
            "relationship_state": data.get("relationship_state") if isinstance(data, dict) else None
        })

    async def _on_images(self, data):
        """Handle images response"""
        await self._emit_callback({
            "type": "images",
            "images": data.get("images", []) if isinstance(data, dict) else []
        })

    async def _on_activity(self, data):
        """Handle activity response"""
        await self._emit_callback({
            "type": "activity",
            "current_status": data.get("current_status") if isinstance(data, dict) else None,
            "activities": data.get("activities", []) if isinstance(data, dict) else []
        })

    async def _on_autopilot(self, data):
        """Handle autopilot status response"""
        await self._emit_callback({
            "type": "autopilot",
            "time": data.get("time") if isinstance(data, dict) else None,
            "activity": data.get("activity") if isinstance(data, dict) else None,
            "location": data.get("location") if isinstance(data, dict) else None,
            "interruptibility": data.get("interruptibility") if isinstance(data, dict) else None,
            "natural_gap": data.get("natural_gap") if isinstance(data, dict) else None,
            "description": data.get("description") if isinstance(data, dict) else None,
            "energy": data.get("energy") if isinstance(data, dict) else None,
        })

    async def _on_fact_approval_request(self, data):
        """Handle fact approval request - prompts user to approve/reject a fact"""
        await self._emit_callback({
            "type": "fact_approval_request",
            "pending_id": data.get("pending_id"),
            "subject": data.get("subject"),
            "fact_text": data.get("fact_text"),
            "category": data.get("category"),
            "sensitivity": data.get("sensitivity"),
            "reason": data.get("reason")
        })

    async def _on_pending_facts(self, data):
        """Handle pending facts list response"""
        await self._emit_callback({
            "type": "pending_facts",
            "count": data.get("count", 0),
            "facts": data.get("facts", [])
        })

    async def _on_fact_approved(self, data):
        """Handle fact approved confirmation"""
        await self._emit_callback({
            "type": "fact_approved",
            "fact_id": data.get("fact_id"),
            "status": data.get("status")
        })

    async def _on_fact_rejected(self, data):
        """Handle fact rejected confirmation"""
        await self._emit_callback({
            "type": "fact_rejected",
            "fact_id": data.get("fact_id"),
            "status": data.get("status")
        })

    async def _on_fact_edited(self, data):
        """Handle fact edited confirmation"""
        await self._emit_callback({
            "type": "fact_edited",
            "fact_id": data.get("fact_id"),
            "status": data.get("status"),
            "new_text": data.get("new_text")
        })

    async def _emit_callback(self, message):
        """Emit message to all registered callbacks"""
        for callback in self.message_callbacks:
            try:
                await callback(message)
            except Exception:
                pass  # Silently ignore callback errors

    async def connect(self) -> bool:
        """
        Connect to backend

        Returns:
            True if connected successfully
        """
        try:
            # Prepare authentication
            auth = {}
            if self.api_key:
                auth['token'] = self.api_key
            else:
                email = os.environ.get("COMPANION_EMAIL", "user@example.com")
                auth['email'] = email

            await self.sio.connect(
                self.backend_url,
                auth=auth if auth else None,
                wait_timeout=10
            )
            return True
        except Exception:
            return False

    async def disconnect(self):
        """Close connection"""
        if self.sio.connected:
            await self.sio.disconnect()
            self.connected = False

    async def send_message(self, message: str, message_type: str = 'chat') -> bool:
        """
        Send message to backend

        Args:
            message: Message text
            message_type: 'chat' (normal) or 'test' (dry run, not saved)

        Returns:
            True if sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('send_message', {
                'message': message,
                'message_type': message_type,
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception:
            self.connected = False
            return False

    async def request_history(self, limit: int = 5) -> bool:
        """
        Request message history from backend

        Args:
            limit: Number of messages to fetch

        Returns:
            True if request sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('request_history', {'limit': limit})
            return True
        except Exception:
            return False

    async def request_state(self) -> bool:
        """
        Request current scene and internal state from backend

        Returns:
            True if request sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('request_state', {})
            return True
        except Exception:
            return False

    async def request_images(self, limit: int = 20) -> bool:
        """
        Request recent generated images from backend

        Args:
            limit: Number of images to fetch

        Returns:
            True if request sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('request_images', {'limit': limit})
            return True
        except Exception:
            return False

    async def request_activity(self) -> bool:
        """
        Request current activity info from backend

        Returns:
            True if request sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('request_activity', {})
            return True
        except Exception:
            return False

    async def request_autopilot(self) -> bool:
        """
        Request James's autopilot status from backend

        Returns:
            True if request sent successfully
        """
        if not self.connected:
            return False

        try:
            await self.sio.emit('request_autopilot', {})
            return True
        except Exception:
            return False

    async def request_pending_facts(self) -> bool:
        """Request list of pending facts awaiting approval"""
        if not self.connected:
            return False
        try:
            await self.sio.emit('request_pending_facts', {})
            return True
        except Exception:
            return False

    async def approve_fact(self, fact_id: int) -> bool:
        """Approve a pending fact"""
        if not self.connected:
            return False
        try:
            await self.sio.emit('approve_fact', {'fact_id': fact_id})
            return True
        except Exception:
            return False

    async def reject_fact(self, fact_id: int, reason: str = "") -> bool:
        """Reject a pending fact"""
        if not self.connected:
            return False
        try:
            await self.sio.emit('reject_fact', {'fact_id': fact_id, 'reason': reason})
            return True
        except Exception:
            return False

    async def edit_fact(self, fact_id: int, new_text: str) -> bool:
        """Edit and approve a pending fact"""
        if not self.connected:
            return False
        try:
            await self.sio.emit('edit_fact', {'fact_id': fact_id, 'fact_text': new_text})
            return True
        except Exception:
            return False

    async def listen(self, callback: Callable):
        """
        Register callback for incoming messages

        Args:
            callback: Async function to call with received messages
        """
        self.message_callbacks.append(callback)

        try:
            while self.connected:
                await asyncio.sleep(1)
        except Exception:
            pass
