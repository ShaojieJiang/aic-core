"""Feedly MCP server."""

from .server import server


def main() -> None:
    """Start the MCP server."""
    server.run()
