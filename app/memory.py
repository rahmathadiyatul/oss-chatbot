from typing import Dict, List

# super simple in-memory session store
_memory: Dict[int, List[dict]] = {}

def get_session(session_id: int) -> List[dict]:
    return _memory.setdefault(session_id, [])

def add_message(session_id: int, role: str, content: str) -> None:
    get_session(session_id).append({"role": role, "content": content})

def clear_session(session_id: int) -> None:
    _memory.pop(session_id, None)
