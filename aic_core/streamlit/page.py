"""Streamlit page class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
from streamlit import session_state


def app_state(file_path: str) -> Callable:
    """Singleton decorator that takes a file path argument."""

    def decorator(cls: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            if file_path not in session_state:
                session_state[file_path] = cls(*args, **kwargs)
            return session_state[file_path]

        return wrapper

    return decorator


class AICPage(ABC):
    """AIC page."""

    @abstractmethod
    def run(self) -> None:
        """Run the page."""
        pass  # pragma: no cover
