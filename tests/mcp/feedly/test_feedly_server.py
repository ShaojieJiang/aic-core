from unittest.mock import Mock, patch
import pytest
from aic_core.mcp.feedly.server import get_feedly_news, read_uninteresting


@pytest.fixture
def mock_feedly_session():
    with patch("aic_core.mcp.feedly.server.FeedlySession") as mock_session:
        # Create mock article data
        mock_articles = [
            {"title": "Test Article 1", "id": "article1"},
            {"title": "Test Article 2", "id": "article2"},
        ]

        # Setup the mock chain
        mock_session_instance = Mock()
        mock_user = Mock()
        mock_category = Mock()

        mock_session.return_value = mock_session_instance
        mock_session_instance.user = mock_user
        mock_user.user_categories = {"AI": mock_category}
        mock_category.stream_contents.return_value = mock_articles

        mock_session_instance.auth.auth_token = "mock_token"
        yield mock_session


def test_get_feedly_news(mock_feedly_session):
    # Call the function
    results = get_feedly_news(max_count=2, category="AI", feedly_token="test_token")

    # Verify the results
    assert len(results) == 2
    assert results[0] == {"title": "Test Article 1", "id": "article1"}
    assert results[1] == {"title": "Test Article 2", "id": "article2"}

    # Verify the mock was called correctly
    mock_feedly_session.assert_called_once_with(auth="test_token")
    mock_session_instance = mock_feedly_session.return_value
    mock_category = mock_session_instance.user.user_categories["AI"]
    mock_category.stream_contents.assert_called_once()


def test_get_feedly_news_empty_results(mock_feedly_session):
    # Modify the mock to return empty results
    mock_session_instance = mock_feedly_session.return_value
    mock_category = mock_session_instance.user.user_categories["AI"]
    mock_category.stream_contents.return_value = []

    # Call the function
    results = get_feedly_news(max_count=2, category="AI", feedly_token="test_token")

    # Verify the results
    assert len(results) == 0


def test_read_uninteresting_success(mock_feedly_session):
    with patch("aic_core.mcp.feedly.server.httpx.post") as mock_post:
        # Setup mock response
        mock_post.return_value.status_code = 200

        # Call the function
        read_uninteresting(entry_id="test_entry_123", feedly_token="test_token")

        # Verify the POST request was made correctly
        mock_post.assert_called_once_with(
            "https://cloud.feedly.com/v3/markers",
            json={
                "action": "markAsRead",
                "type": "entries",
                "entryIds": ["test_entry_123"],
            },
            headers={
                "Authorization": "OAuth mock_token",
                "Content-Type": "application/json",
            },
        )


def test_read_uninteresting_failure():
    with patch("aic_core.mcp.feedly.server.httpx.post") as mock_post:
        # Setup mock response for failure case
        mock_post.return_value.status_code = 400

        # Call the function
        read_uninteresting(entry_id="test_entry_123", feedly_token="test_token")

        # Verify the POST request was made
        mock_post.assert_called_once()

        # The function should not raise an exception, but log an error
        # You might want to add logger verification if needed
