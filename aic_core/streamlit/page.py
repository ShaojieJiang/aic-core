"""Streamlit page class."""

from abc import ABC
from abc import abstractmethod
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


def app_state_registry(cls: Callable, file_path: str) -> Any:
    """Create or return a singleton instance of a class.

    Args:
        cls: The class to instantiate
        file_path: Unique identifier for the instance

    Returns:
        Instance of the class
    """
    if file_path not in session_state:
        session_state[file_path] = cls()
    return session_state[file_path]


class AICPage(ABC):
    """AIC page."""

    @abstractmethod
    def run(self) -> None:
        """Run the page."""
        pass  # pragma: no cover
