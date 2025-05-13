import os
from typing import Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelRequest, RetryPromptPart, TextPart
from aic_core.agent.agent import AgentConfig, AgentFactory, AICAgent, MCPServerStdio
from aic_core.agent.result_types import TableOutput


def test_agent_config_initialization():
    """Test basic initialization of AgentConfig."""
    config = AgentConfig(
        model="openai:gpt-4o",
        name="TestAgent",
        system_prompt="Test prompt",
        retries=2,
        repo_id="test-repo",
    )

    assert config.model == "openai:gpt-4o"
    assert config.name == "TestAgent"
    assert config.system_prompt == "Test prompt"
    assert config.retries == 2
    assert config.hf_tools == []
    assert config.mcp_servers == []


@patch("aic_core.agent.agent.AgentHub")
def test_from_hub(mock_agent_hub):
    """Test loading config from Hugging Face Hub."""
    # Create a Mock instance for the repo
    mock_repo = Mock()
    mock_agent_hub.return_value = mock_repo

    # Set up the load_config mock on the repo instance
    mock_repo.load_config.return_value = {
        "model": "openai:gpt-4o",
        "name": "TestAgent",
        "system_prompt": "Test prompt",
        "repo_id": "test-repo",
    }

    config = AgentConfig.from_hub("test-repo", "agent")

    assert config.model == "openai:gpt-4o"
    assert config.name == "TestAgent"
    assert config.system_prompt == "Test prompt"
    mock_repo.load_config.assert_called_with("agent")


@patch("aic_core.agent.agent.AgentHub")
def test_push_to_hub(mock_agent_hub):
    """Test pushing config to Hugging Face Hub."""
    mock_repo = Mock()
    mock_agent_hub.return_value = mock_repo

    config = AgentConfig(model="openai:gpt-4o", name="TestAgent", repo_id="test-repo")

    config.push_to_hub()

    mock_repo.upload_content.assert_called_once()


@pytest.fixture
def basic_config():
    return AgentConfig(
        model="openai:gpt-4o",
        name="TestAgent",
        result_type=["str"],
        known_tools=["tool1"],
        hf_tools=["tool2"],
        mcp_servers=["command1 arg1 arg2", "command2"],
        repo_id="test-repo",
    )


@pytest.fixture
def agent_factory(basic_config):
    return AgentFactory(basic_config)


def test_init(basic_config):
    factory = AgentFactory(basic_config)
    assert factory.config == basic_config


@patch("aic_core.agent.agent.load_tool")
def test_hf_to_pai_tools_local(mock_load_tool):
    # Setup mock tool
    def forward():
        """Test tool docstring"""
        pass

    mock_tool = Mock()
    mock_tool.forward = forward
    mock_tool.name = "test_tool"
    mock_tool.description = "test description"
    mock_load_tool.return_value = mock_tool

    tool = AgentFactory.hf_to_pai_tools("test_tool")
    assert isinstance(tool, Tool)
    assert tool.name == "test_tool"

    # Verify local_files_only was tried first
    mock_load_tool.assert_called_with(
        "test_tool", trust_remote_code=True, local_files_only=True
    )


@patch("aic_core.agent.agent.load_tool")
def test_hf_to_pai_tools_remote(mock_load_tool):
    # First call raises LocalEntryNotFoundError, second call succeeds
    def forward():
        """Test tool docstring"""
        pass

    mock_load_tool.side_effect = [
        LocalEntryNotFoundError("Not found locally"),
        Mock(
            forward=forward,
            name="test_tool",
            description="test description",
            forward__doc__=None,
        ),
    ]

    tool = AgentFactory.hf_to_pai_tools("test_tool")
    assert isinstance(tool, Tool)
    assert mock_load_tool.call_count == 2


def test_get_result_type_empty(agent_factory):
    agent_factory.config.result_type = []
    assert isinstance(agent_factory.get_result_type(), type(str))


def test_get_result_type_single(agent_factory):
    agent_factory.config.result_type = ["str"]
    assert isinstance(agent_factory.get_result_type(), type(str))


@patch("aic_core.agent.agent.AgentHub")
def test_get_result_type_structured_output(mock_agent_hub, agent_factory):
    # Setup mock for AgentHub
    mock_repo = Mock()
    mock_custom_type = type("CustomType", (), {})  # Creates a dynamic type
    mock_repo.load_result_type.return_value = mock_custom_type
    mock_agent_hub.return_value = mock_repo

    # Test with a mix of built-in and custom types
    agent_factory.config.result_type = ["str", "custom_output_type"]
    result = agent_factory.get_result_type()

    # Verify AgentHub was called for custom type
    mock_repo.load_result_type.assert_called_once_with("custom_output_type")

    # Verify the resulting Union type contains both str and our custom type
    assert result == Union.__getitem__((str, mock_custom_type))


def test_get_result_type_known_type(agent_factory):
    agent_factory.config.result_type = ["TableOutput"]
    result = agent_factory.get_result_type()
    assert result == TableOutput


@patch("aic_core.agent.agent.AgentHub")
def test_get_tools(mock_agent_hub, agent_factory):
    # Setup mocks
    def example_tool():
        """Example tool docstring"""
        pass

    mock_repo = Mock()
    mock_repo.load_tool.return_value = example_tool
    mock_agent_hub.return_value = mock_repo

    with patch.object(AgentFactory, "hf_to_pai_tools") as mock_hf_to_pai:
        mock_hf_to_pai.return_value = Tool(example_tool, name="hf_tool")
        tools = agent_factory.get_tools()

        assert len(tools) == 2  # One known_tool and one hf_tool
        assert all(isinstance(tool, Tool) for tool in tools)


def test_get_mcp_servers(agent_factory):
    servers = agent_factory.get_mcp_servers()
    assert len(servers) == 2
    assert servers[0] == MCPServerStdio("command1", ["arg1", "arg2"])
    assert servers[1] == MCPServerStdio("command2", [])


@patch("aic_core.agent.agent.OpenAIProvider")
def test_create_agent(mock_provider, agent_factory):
    # Setup mocks
    mock_provider_instance = Mock()
    mock_provider.return_value = mock_provider_instance

    with (
        patch.object(AgentFactory, "get_tools") as mock_get_tools,
        patch.object(AgentFactory, "get_result_type") as mock_get_result_type,
        patch.object(AgentFactory, "get_mcp_servers") as mock_get_mcp_servers,
    ):
        mock_get_tools.return_value = []
        mock_get_result_type.return_value = str
        mock_get_mcp_servers.return_value = []

        agent = agent_factory.create_agent("test-api-key")

        assert isinstance(agent, Agent)
        mock_provider.assert_called_once_with(api_key="test-api-key")


def test_agent_with_logfire():
    """Test agent initialization with logfire."""
    with patch.dict(os.environ, {"LOGFIRE_TOKEN": "test-token"}):
        with patch("logfire.configure") as mock_configure:
            with patch("logfire.instrument_pydantic_ai") as mock_instrument:
                # Re-import to trigger the logfire configuration
                from importlib import reload
                import aic_core.agent.agent

                reload(aic_core.agent.agent)

                mock_configure.assert_called_once()
                mock_instrument.assert_called_once()


@pytest.mark.asyncio
async def test_aic_agent_skip_retry_msgs():
    """Test AICAgent with retry messages."""
    # Create mock messages with proper parts
    normal_msg = ModelRequest(parts=[TextPart(content="Hello")])
    retry_prompt = ModelRequest(parts=[RetryPromptPart(content="Retry")])
    retry_result = ModelRequest(parts=[TextPart(content="Retry result")])
    final_msg = ModelRequest(parts=[TextPart(content="Final")])

    mock_result = MagicMock()
    # When skip_retry_msgs=True, retry_prompt and retry_result should be filtered out
    mock_result.new_messages.return_value = [
        normal_msg,
        retry_prompt,
        retry_result,
        final_msg,
    ]

    # Create mock agent
    mock_agent = MagicMock()
    mock_agent._mcp_servers = []
    mock_agent.run = AsyncMock(return_value=mock_result)

    # Mock AgentHub and its dependencies
    mock_hub = MagicMock()
    mock_hub.load_config.return_value = {
        "model": "openai:gpt-4",
        "repo_id": "test-repo",
        "name": "test-agent",
    }

    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        patch("aic_core.agent.agent.AgentHub", return_value=mock_hub),
        patch("pydantic_ai.models.openai.OpenAIModel") as mock_model_cls,
        patch("pydantic_ai.providers.openai.OpenAIProvider") as mock_provider_cls,
        patch("openai.AsyncOpenAI") as mock_openai_cls,
    ):
        # Mock OpenAI model and provider
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_openai = MagicMock()
        mock_provider_cls.return_value = mock_provider
        mock_model_cls.return_value = mock_model
        mock_openai_cls.return_value = mock_openai

        # Create AICAgent instance
        aic_agent = AICAgent("test-repo", "test-agent")
        aic_agent.agent = mock_agent

        # Test with skip_retry_msgs=True
        result = await aic_agent.get_response("test", [], skip_retry_msgs=True)
        assert len(result) == 3
        assert result[0].parts[0].content == "Hello"
        assert result[-1].parts[0].content == "Final"

        # Test with skip_retry_msgs=False
        result = await aic_agent.get_response("test", [], skip_retry_msgs=False)
        assert len(result) == 4
        assert result[0].parts[0].content == "Hello"
        assert result[1].parts[0].content == "Retry"
        assert result[2].parts[0].content == "Retry result"
        assert result[3].parts[0].content == "Final"


def test_agent_factory_model_settings():
    """Test AgentFactory with model settings."""
    config = AgentConfig(
        model="openai:gpt-4",
        name="TestAgent",
        model_settings={"temperature": 0.7},
        repo_id="test-repo",
    )
    factory = AgentFactory(config)

    # Mock OpenAI dependencies
    mock_provider = MagicMock()
    mock_model = MagicMock()

    with patch(
        "pydantic_ai.providers.openai.OpenAIProvider", return_value=mock_provider
    ):
        with patch("pydantic_ai.models.openai.OpenAIModel", return_value=mock_model):
            with patch("openai.AsyncOpenAI", return_value=MagicMock()):
                # Create the agent
                agent = factory.create_agent("test-key")

                # Verify the agent was created with the correct settings
                assert agent.name == "TestAgent"
                assert agent.model_settings == {"temperature": 0.7}


def test_agent_factory_empty_mcp_server():
    """Test AgentFactory with empty MCP server."""
    config = AgentConfig(
        model="openai:gpt-4",
        name="TestAgent",
        mcp_servers=["  ", "command arg1 arg2"],
        repo_id="test-repo",
    )
    factory = AgentFactory(config)
    servers = factory.get_mcp_servers()
    assert len(servers) == 1
    assert servers[0].command == "command"
    assert servers[0].args == ["arg1", "arg2"]
