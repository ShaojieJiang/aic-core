import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
from aic_core.agent.agent import AgentConfig, AgentFactory
from aic_core.streamlit.agent_page import AgentPage, PageState


@pytest.fixture
def agent_page():
    return AgentPage(repo_id="test-repo", page_state=PageState())


@pytest.fixture
def mock_agent():
    agent = MagicMock(spec=Agent)
    agent._mcp_servers = []
    return agent


def test_init(agent_page):
    assert agent_page.repo_id == "test-repo"
    assert agent_page.page_title == "Agent"
    assert agent_page.user_role == "user"
    assert agent_page.assistant_role == "assistant"


def test_reset_chat_history(agent_page):
    agent_page.page_state.chat_history = ["some", "messages"]
    agent_page.reset_chat_history()
    assert agent_page.page_state.chat_history == []


def test_get_agent():
    # Setup
    agent_page = AgentPage(repo_id="test-repo", page_state=PageState())
    mock_agent = MagicMock(spec=Agent)
    mock_config = MagicMock(spec=AgentConfig)

    # Mock the from_hub method of AgentConfig
    with patch.object(
        AgentConfig, "from_hub", return_value=mock_config
    ) as mock_from_hub:
        # Mock the AgentFactory
        with patch.object(
            AgentFactory, "create_agent", return_value=mock_agent
        ) as mock_create_agent:
            # Call the method
            result = agent_page.get_agent("test-agent")

            # Verify the calls
            mock_from_hub.assert_called_once_with("test-repo", "test-agent")
            mock_create_agent.assert_called_once()

            # Verify the result
            assert result == mock_agent


def test_get_response_without_mcp_servers(agent_page, mock_agent):
    user_input = "Hello"
    mock_result = MagicMock()
    mock_result.new_messages.return_value = ["message1", "message2"]
    mock_agent.run = AsyncMock(return_value=mock_result)

    with patch("streamlit.chat_message") as mock_chat_message:
        asyncio.run(agent_page.get_response(user_input, mock_agent))

        mock_chat_message.assert_called_once()
        assert agent_page.page_state.chat_history == ["message1", "message2"]


def test_get_response_with_mcp_servers(agent_page, mock_agent):
    mock_agent._mcp_servers = ["server1"]
    user_input = "Hello"
    mock_result = MagicMock()
    mock_result.new_messages.return_value = ["message1"]
    mock_agent.run = AsyncMock(return_value=mock_result)
    mock_agent.run_mcp_servers = MagicMock()
    mock_agent.run_mcp_servers.return_value.__aenter__ = AsyncMock()
    mock_agent.run_mcp_servers.return_value.__aexit__ = AsyncMock()
    agent_page.reset_chat_history()

    with patch("streamlit.chat_message") as mock_chat_message:
        asyncio.run(agent_page.get_response(user_input, mock_agent))

        mock_chat_message.assert_called_once()
        assert agent_page.page_state.chat_history == ["message1"]


def test_display_chat_history(agent_page):
    message = ModelRequest(
        parts=[TextPart(content="Hello"), UserPromptPart(content="Hi")]
    )
    agent_page.page_state.chat_history = [message]

    with patch("streamlit.chat_message") as mock_chat_message:
        agent_page.display_chat_history()
        assert mock_chat_message.call_count == 2


@patch("streamlit.title")
@patch("streamlit.chat_input")
@patch("streamlit.sidebar.button")
def test_run(mock_button, mock_chat_input, mock_title, agent_page):
    mock_chat_input.return_value = None
    agent_page.agent_selector = MagicMock()
    agent_page.get_agent = MagicMock()

    with patch.object(agent_page, "display_chat_history") as mock_display_chat_history:
        agent_page.run()

        mock_title.assert_called_once_with("Agent")
        mock_button.assert_called_once_with(
            "Reset chat history", on_click=agent_page.reset_chat_history
        )
        mock_chat_input.assert_called_once_with("Enter a message")
        mock_display_chat_history.assert_called_once()
