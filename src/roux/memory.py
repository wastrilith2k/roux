from typing import List, Dict, Optional
from datetime import datetime
import uuid


class Episode:
  """A single event or message in a session"""
  def __init__(
    self,
    session_id: str,
    user_id:str,
    content: str,
    role: Optional[str] = None,
    event_type: str = 'message',
    metadata: Optional[Dict] = None
  ):
    self.id = str(uuid.uuid4())
    self.session_id = session_id
    self.user_id = user_id
    self.content = content
    self.role = role
    self.event_type = event_type
    self.timestamp = datetime.now().isoformat()
    self.metadata = metadata or {}

  def to_dict(self) -> Dict:
    return {
      "id": self.id,
      "session_id": self.session_id,
      "user_id": self.user_id,
      "content": self.content,
      "role": self.role,
      "event_type": self.event_type,
      "timestamp": self.timestamp,
      "metadata": self.metadata
    }

class Session:
  """A conversation or game session with multiple participants"""
  def __init__(self, session_id: Optional[str] = None, participants: Optional[List[str]] = None):
    self.id = session_id or str(uuid.uuid4())
    self.participants = participants or []
    self.episodes: List[Episode] = []
    self.created_at = datetime.now().isoformat()

  def add_particpant(self, user_id: str):
    """Add a participant to this session"""
    if user_id not in self.participants:
      self.participants.append(user_id)

  def add_message(
    self,
    user_id: str,
    content: str,
    role: Optional[str] = None,
    metadata: Optional[Dict] = None
  ) -> Episode:
    """Add a message from a participant"""
    episode = Episode(
      session_id=self.id,
      user_id=user_id,
      content=content,
      role=role,
      event_type='message',
      metadata=metadata
    )
    self.episodes.append(episode)
    return episode

  def add_event(
      self,
      user_id: str,
      event_type: str,
      data: Dict
  ) -> Episode:
    """Add a non-message event (game action, etc)"""
    episode = Episode(
      session_id=self.id,
      user_id=user_id,
      content=str(data),
      event_type=event_type,
      metadata=data
    )
    self.episodes.append(episode)
    return episode

  def get_episodes(
    self,
    limit: Optional[int] = None,
    user_id: Optional[str] = None,
    event_type: Optional[str] = None
  ) -> List[Episode]:
    """
    Get episodes with optional filtering

    Args:
      limit: Max number to return
      user_id: Filter by specific user
      event_type: Filter by specific event type
    """

    episodes = self.episodes

    if user_id:
      episodes = [ep for ep in episodes if ep.user_id == user_id]

    if event_type:
      episodes = [ep for ep in episodes if ep.event_type == event_type]

    if limit:
      episodes = episodes[-limit:]

    return episodes

  def get_messages_for_llm(self, limit: int = 10, include_user_labels: bool = False) -> List[Dict]:
    """
    Get messages in LLM format

    Args:
      limit: Max messages to return
      include_user_labels: If True and multiple parpticipants, prefix with [user_id]
    """

    message_episodes = [ep for ep in self.episodes if ep.event_type == 'message']
    recent = message_episodes[-limit:]

    # If multiple participants, show who said what
    show_labels = include_user_labels and len(self.participants) > 2

    return [
      {
        'role': ep.role or 'user',
        'content': f"[{ep.user_id}] {ep.content}" if show_labels else ep.content
      }
      for ep in recent
    ]

class Memory:
  """Main memory interface - manages sessions"""

  def __init__(self):
    self.sessions: Dict[str, Session] = {}
    self.current_session_id: Optional[str] = None

  def create_session(
      self,
      session_id: Optional[str] = None,
      participants: Optional[List[str]] = None
  ) -> Session:
    """Create a new session"""
    session = Session(session_id=session_id, participants=participants)
    self.sessions[session.id] = session
    self.current_session_id = session.id
    return session

  def get_session(
      self,
      session_id:str
  ) -> Optional[Session]:
    """Retrieve a session by ID"""
    return self.sessions.get(session_id)

  def get_current_session(
      self
  ) -> Optional[Session]:
    """Get the current active session"""
    if self.current_session_id:
      return self.sessions.get(self.current_session_id)
    return None

  def set_current_session(self, session_id:str):
    """Set the current active session"""
    if session_id in self.sessions:
      self.current_session_id = session_id
    else:
      raise ValueError(f"Session ID {session_id} does not exist.")

  # Convenience methods for current session
  def add_message(
    self,
    user_id: str,
    content: str,
    **kwargs
  ):
    """Add a message to the current session"""
    session = self.get_current_session()
    if not session:
      session = self.create_session(participants=[user_id])
    return session.add_message(user_id=user_id, content=content, **kwargs)

  def get_context_for_llm(self, limit: int = 10) -> List[Dict]:
    """Get context from current session"""
    session = self.get_current_session()
    if session:
      return session.get_messages_for_llm(limit)
    return []