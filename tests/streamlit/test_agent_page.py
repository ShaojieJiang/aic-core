import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from aic_core.agent.agent import AICAgent
from aic_core.agent.result_types import ComponentRegistry
from aic_core.streamlit.agent_page import AgentPage, PageState


@pytest.fixture
def agent_page():
    """Create an agent page fixture."""
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


def test_get_response_without_mcp_servers(agent_page, mock_agent):
    """Test get_response method without MCP servers."""
    user_input = "Hello"
    mock_result = MagicMock()
    mock_result.new_messages.return_value = ["message1", "message2"]
    mock_agent.run = AsyncMock(return_value=mock_result)
    mock_agent.get_response = AsyncMock()
    agent_page.agent = mock_agent
    agent_page.page_state.chat_history = []

    asyncio.run(agent_page.get_response(user_input))
    mock_agent.get_response.assert_called_once_with(
        user_input, agent_page.page_state.chat_history
    )


def test_get_response_with_mcp_servers(agent_page, mock_agent):
    """Test get_response method with MCP servers."""
    mock_agent._mcp_servers = ["server1"]
    user_input = "Hello"
    mock_result = MagicMock()
    mock_result.new_messages.return_value = ["message1"]
    mock_agent.run = AsyncMock(return_value=mock_result)
    mock_agent.get_response = AsyncMock()
    mock_agent.run_mcp_servers = MagicMock()
    mock_agent.run_mcp_servers.return_value.__aenter__ = AsyncMock()
    mock_agent.run_mcp_servers.return_value.__aexit__ = AsyncMock()
    agent_page.reset_chat_history()
    agent_page.agent = mock_agent

    asyncio.run(agent_page.get_response(user_input))
    mock_agent.get_response.assert_called_once_with(
        user_input, agent_page.page_state.chat_history
    )


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
    """Test run method."""
    mock_chat_input.return_value = None
    agent_page.agent_selector = MagicMock()
    agent_page.agent_selector.return_value = None

    with (
        patch.object(agent_page, "display_chat_history") as mock_display_chat_history,
        patch.object(AICAgent, "__init__", return_value=None) as mock_init,
    ):
        agent_page.run()

        mock_button.assert_called_once_with(
            "Reset chat history", on_click=agent_page.reset_chat_history
        )
        mock_chat_input.assert_called_once_with("Enter a message")
        mock_display_chat_history.assert_called_once()
        mock_init.assert_called_once()


@patch("streamlit.title")
@patch("streamlit.chat_input")
@patch("streamlit.sidebar.button")
def test_run_with_input(mock_button, mock_chat_input, mock_title, agent_page):
    """Test run method with user input."""
    mock_chat_input.return_value = "test input"
    agent_page.agent_selector = MagicMock()
    agent_page.agent_selector.return_value = None
    agent_page.get_response = AsyncMock()

    with (
        patch.object(agent_page, "display_chat_history") as mock_display_chat_history,
        patch("streamlit.rerun") as mock_rerun,
        patch.object(AICAgent, "__init__", return_value=None) as mock_init,
    ):
        agent_page.run()

        mock_button.assert_called_once_with(
            "Reset chat history", on_click=agent_page.reset_chat_history
        )
        mock_chat_input.assert_called_once_with("Enter a message")
        mock_display_chat_history.assert_called_once()
        mock_init.assert_called_once()
        mock_rerun.assert_called_once()


def test_display_parts(agent_page):
    """Test display_parts method with different message parts."""
    # Mock streamlit components
    mock_chat_message = MagicMock()
    mock_chat_message.return_value.write = MagicMock()

    # Test TextPart
    with patch("streamlit.chat_message", return_value=mock_chat_message):
        agent_page.display_parts([TextPart(content="Hello")], None)

    # Reset mock for next test
    mock_chat_message.reset_mock()

    # Test UserPromptPart
    with patch("streamlit.chat_message", return_value=mock_chat_message):
        agent_page.display_parts([UserPromptPart(content="Hi")], None)

    # Reset mock for next test
    mock_chat_message.reset_mock()

    # Test ToolCallPart with valid component
    with patch("streamlit.chat_message", return_value=mock_chat_message):
        with patch.object(ComponentRegistry, "contains_component", return_value=True):
            with patch.object(
                ComponentRegistry, "generate_st_component"
            ) as mock_generate:
                tool_call = ToolCallPart(
                    tool_name="test_tool",
                    args="{}",
                    tool_call_id="123",
                    part_kind="tool-call",
                )
                tool_return = ToolReturnPart(
                    tool_name="test_tool",
                    content="result",
                    tool_call_id="123",
                    part_kind="tool-return",
                )
                agent_page.display_parts([tool_call], tool_return)
                mock_generate.assert_called_once_with(
                    tool_call, tool_return, agent_page.input_callback
                )

    # Reset mock for next test
    mock_chat_message.reset_mock()

    # Test ToolCallPart with invalid component
    with patch("streamlit.chat_message", return_value=mock_chat_message):
        with patch.object(ComponentRegistry, "contains_component", return_value=False):
            tool_call = ToolCallPart(
                tool_name="invalid_tool",
                args="{}",
                tool_call_id="123",
                part_kind="tool-call",
            )
            agent_page.display_parts([tool_call], None)


def test_get_response_without_agent(agent_page):
    """Test get_response without agent initialized."""
    with pytest.raises(AssertionError):
        asyncio.run(agent_page.get_response("test"))


def test_input_callback(agent_page, mock_agent):
    """Test input callback."""
    agent_page.agent = mock_agent
    agent_page.get_response = AsyncMock()

    # Mock session state
    with patch("streamlit.session_state", {"test_key": "test_value"}):
        tool_call = ToolCallPart(
            tool_name="test_tool",
            args=json.dumps({"label": "Test Label"}),
            tool_call_id="123",
            part_kind="tool-call",
        )
        tool_return = ToolReturnPart(
            tool_name="test_tool",
            content="result",
            tool_call_id="123",
            part_kind="tool-return",
        )

        agent_page.input_callback("test_key", tool_call, tool_return)

        # Verify the tool return content was updated
        assert tool_return.content == "User input: test_value"
        # Verify get_response was called with the correct message
        agent_page.get_response.assert_called_once_with(
            "My answer to 'Test Label' is: test_value",
            manual_answer=False,
        )
