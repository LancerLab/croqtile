#!/usr/bin/env python3
"""
Session Logger Module
--------------------
This module provides functionality to log agent interactions in the Pinocchio system.
It records the full history of a session, including agent activations, inputs, outputs,
and context updates.
"""

import os
import sys
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'session_logger.log'))
    ]
)
logger = logging.getLogger('session_logger')

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
PINOCCHIO_DIR = PROJECT_ROOT / "tools" / "pinocchio"
SESSIONS_DIR = PINOCCHIO_DIR / ".cursor" / "state" / "sessions"

class SessionLogger:
    """Class to log agent interactions in a session."""
    
    def __init__(self, session_id=None):
        """Initialize the session logger."""
        self.session_id = session_id or self._generate_session_id()
        self.start_time = int(time.time())
        self.end_time = None
        self.user_request = None
        self.agent_interactions = []
        self.final_result = {}
        self.session_file = SESSIONS_DIR / f"{self.session_id}.json"
        
        # Create sessions directory if it doesn't exist
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Session logger initialized with ID: {self.session_id}")
    
    def _generate_session_id(self):
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"session-{timestamp}-{unique_id}"
    
    def set_user_request(self, request):
        """Set the initial user request."""
        self.user_request = request
        logger.info(f"User request set: {request[:50]}...")
        self._save_session()
    
    def log_interaction(self, agent, input_text, output_text, context_updates=None):
        """Log an agent interaction."""
        interaction = {
            "agent": agent,
            "timestamp": int(time.time()),
            "input": input_text,
            "output": output_text,
            "context_updates": context_updates or {}
        }
        
        self.agent_interactions.append(interaction)
        logger.info(f"Logged interaction for agent: {agent}")
        self._save_session()
        
        return interaction
    
    def set_final_result(self, result):
        """Set the final result of the session."""
        self.final_result = result
        self.end_time = int(time.time())
        logger.info("Final result set, session completed")
        self._save_session()
    
    def _save_session(self):
        """Save the session to a file."""
        session_data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "user_request": self.user_request,
            "agent_interactions": self.agent_interactions,
            "final_result": self.final_result
        }
        
        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session saved to {self.session_file}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    @classmethod
    def load_session(cls, session_id):
        """Load a session from a file."""
        session_file = SESSIONS_DIR / f"{session_id}.json"
        
        if not session_file.exists():
            logger.error(f"Session file not found: {session_file}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.info(f"Session loaded: {session_id}")
            
            # Create a new session logger and populate it with the loaded data
            session_logger = cls(session_id=session_id)
            session_logger.start_time = session_data.get("start_time")
            session_logger.end_time = session_data.get("end_time")
            session_logger.user_request = session_data.get("user_request")
            session_logger.agent_interactions = session_data.get("agent_interactions", [])
            session_logger.final_result = session_data.get("final_result", {})
            
            return session_logger
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None
    
    @classmethod
    def list_sessions(cls, limit=10):
        """List recent sessions."""
        if not SESSIONS_DIR.exists():
            logger.warning(f"Sessions directory not found: {SESSIONS_DIR}")
            return []
        
        try:
            session_files = list(SESSIONS_DIR.glob("*.json"))
            session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            sessions = []
            for file in session_files[:limit]:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    sessions.append({
                        "session_id": session_data.get("session_id"),
                        "start_time": session_data.get("start_time"),
                        "end_time": session_data.get("end_time"),
                        "user_request": session_data.get("user_request"),
                        "num_interactions": len(session_data.get("agent_interactions", []))
                    })
                except Exception as e:
                    logger.error(f"Error loading session file {file}: {e}")
            
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

def get_current_session():
    """Get or create the current session logger."""
    # This is a simple implementation that creates a new session each time
    # In a real implementation, you might want to reuse the current session
    return SessionLogger()

def main():
    """Main function for testing."""
    print("Session Logger Test")
    
    # Create a new session
    session = SessionLogger()
    print(f"Created session: {session.session_id}")
    
    # Set user request
    session.set_user_request("Test request")
    
    # Log some interactions
    session.log_interaction("coordinator", "Test input", "Test output")
    session.log_interaction("analyzer", "Analyze this", "Analysis result")
    
    # Set final result
    session.set_final_result({"status": "success"})
    
    # List sessions
    sessions = SessionLogger.list_sessions()
    print(f"Found {len(sessions)} sessions")
    for s in sessions:
        print(f"  {s['session_id']}: {s['user_request']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 