"""Feedly MCP server."""

from .server import server


def main() -> None:  # pragma: no cover
    """Start the MCP server."""
    server.run()
