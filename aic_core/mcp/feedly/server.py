"""Feedly MCP server."""

import httpx
from feedly.api_client.session import FeedlySession
from feedly.api_client.stream import StreamOptions
from mcp.server.fastmcp import FastMCP
from aic_core.logging import get_logger


server = FastMCP("Feedly MCP")
logger = get_logger(__name__)


@server.tool()
def get_feedly_news(
    max_count: int = 10, category: str = "AI", feedly_token: str | None = None
) -> list[dict]:
    """Get the latest news from Feedly.

    Args:
        max_count: The maximum number of news to get.
        category: The category of the news to get.
        feedly_token: The Feedly token to use.
    """
    session = FeedlySession(auth=feedly_token)
    category_name = session.user.user_categories.get(category)
    results = []

    for article in category_name.stream_contents(
        options=StreamOptions(max_count=max_count)
    ):
        results.append(
            {
                "title": article["title"],
                "id": article["id"],
            }
        )

    return results


@server.tool()
def mark_as_read(entry_id: str, feedly_token: str | None = None) -> None:
    """Mark an entry as read.

    Args:
        entry_id: The id of the entry to mark as read.
        feedly_token: The Feedly token to use.
    """
    session = FeedlySession(auth=feedly_token)
    access_token = session.auth.auth_token
    url = "https://cloud.feedly.com/v3/markers"

    headers = {
        "Authorization": f"OAuth {access_token}",
        "Content-Type": "application/json",
    }

    payload = {"action": "markAsRead", "type": "entries", "entryIds": [entry_id]}

    response = httpx.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        logger.info("Entry marked as read successfully.")
    else:
        logger.error(
            f"Failed to mark entry as read. Status code: {response.status_code}"
        )
    logger.info(f"Marking entry {entry_id} as read.")
